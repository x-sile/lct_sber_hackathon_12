import pandas as pd
import numpy as np
import datetime
import logging
from sklearn.metrics import roc_auc_score
from ptls.preprocessing import PandasDataPreprocessor
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing.iterable_seq_len_limit import ISeqLenLimit
import gc
import torch
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAUROC
from ptls.nn import TrxEncoder, RnnSeqEncoder, Head, TransformerSeqEncoder
from functools import partial
from ptls.frames.supervised import SeqToTargetDataset, SequenceToTarget
from ptls.frames import PtlsDataModule


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
pl.seed_everything(1)

cols_emb_dim = {
    "amount": 8,
    "event_type": 8,
    "event_subtype": 8,
    "currency": 3,
    "src_type11": 8,
    "src_type12": 8,
    "dst_type11": 8,
    "dst_type12": 8,
    "src_type21": 8,
    "src_type22": 8,
    "src_type31": 8,
    "src_type32": 8,
    "weekday": 3,
    "hour": 4,
    "hist_target": 8,
    "dt": 3,
    "till_dt": 8,
    "geohash_4": 8,
}


def target_bit_encoding(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


def end_of_next_month(x):
    res = datetime.datetime.strptime(x, "%Y-%m-%d")
    res = datetime.datetime(res.year + int(res.month / 12), ((res.month % 12) + 1), 1)
    res = datetime.datetime(res.year + int(res.month / 12), ((res.month % 12) + 1), 1)
    res = res - datetime.timedelta(days=1)
    return res.strftime("%Y-%m-%d")


logging.info("loading target history")
tr_tgt = pd.read_parquet("data/train_target.parquet")
tr_tgt["target"] = tr_tgt[["target_1", "target_2", "target_3", "target_4"]].apply(list, axis=1)
tr_tgt["hist_target"] = tr_tgt[["target_1", "target_2", "target_3", "target_4"]].apply(target_bit_encoding, axis=1)
te_tgt = pd.read_parquet("data/test_target_b.parquet")
te_tgt["target"] = te_tgt[["target_1", "target_2", "target_3", "target_4"]].apply(list, axis=1)
te_tgt["hist_target"] = te_tgt[["target_1", "target_2", "target_3", "target_4"]].apply(target_bit_encoding, axis=1)
end_of_next_month_dict = dict(zip(range(1, 13), sorted(tr_tgt.mon.unique())))

logging.info("calculating top geohash_4 for each client each month")
geo = pd.concat(
    [
        pd.read_parquet("data/geo_train.parquet")[["client_id", "event_time", "geohash_4"]],
        pd.read_parquet("data/geo_test.parquet")[["client_id", "event_time", "geohash_4"]],
    ]
)
geo["event_time"] = geo["event_time"].dt.month
geo.groupby(["client_id", "event_time"]).geohash_4.value_counts().reset_index()
geo["rn"] = geo.groupby(["client_id", "event_time"]).geohash_4.cumcount()
geo = geo[geo["rn"] == 0][["client_id", "event_time", "geohash_4"]].rename(columns={"event_time": "month"})
top_geos = geo.geohash_4.value_counts().head(64).index.values
geo.loc[~geo.geohash_4.isin(top_geos), "geohash_4"] = -999999

logging.info("loading transactions")
tr_trx = pd.read_parquet("data/trx_train.parquet")
te_trx = pd.read_parquet("data/trx_test.parquet")
tr_trx = tr_trx[tr_trx.client_id.isin(tr_tgt.client_id)]
te_trx = te_trx[te_trx.client_id.isin(te_tgt.client_id)]
tr_trx["event_date"] = tr_trx.event_time.dt.date.astype(str)
te_trx["event_date"] = te_trx.event_time.dt.date.astype(str)
tr_trx["mon"] = tr_trx.event_time.dt.month.map(end_of_next_month_dict)
te_trx["mon"] = te_trx.event_time.dt.month.map(end_of_next_month_dict)
for i in ["src_type12", "dst_type12", "src_type21", "src_type31"]:
    top = tr_trx[i].value_counts().head(64).index.values
    tr_trx.loc[~tr_trx[i].isin(top), i] = -999999
    te_trx.loc[~te_trx[i].isin(top), i] = -999999


def prepare_sample(dt, trx, target_hist, geo, target=None):
    tmp = trx[trx.event_date <= dt]
    if target is not None:
        tmp = tmp.merge(target[["client_id", "target"]], on="client_id")
    target_hist = target_hist[["client_id", "mon", "hist_target"]]
    tmp = tmp.merge(target_hist, on=["client_id", "mon"], how="left")
    tmp["month"] = tmp.event_time.dt.month
    tmp = tmp.merge(geo, on=["client_id", "month"], how="left")
    tmp["dt"] = dt
    tmp.drop(columns=["month"], inplace=True)
    return tmp


logging.info("constructing sample")
smpl = []
for dt in ["2022-10-31", "2022-11-30", "2022-12-31"]:
    tmp = prepare_sample(
        dt,
        tr_trx[tr_trx.event_date <= dt],
        tr_tgt[tr_tgt.mon <= dt],
        geo,
        tr_tgt[tr_tgt.mon == end_of_next_month(dt)],
    )
    logging.info(f"added {tmp.client_id.nunique()} train clients @ {dt} with target")
    smpl.append(tmp)

for dt in ["2022-10-31", "2022-11-30"]:
    tmp = prepare_sample(
        dt,
        te_trx[te_trx.event_date <= dt],
        te_tgt[te_tgt.mon <= dt],
        geo,
        te_tgt[te_tgt.mon == end_of_next_month(dt)],
    )
    logging.info(f"added {tmp.client_id.nunique()} test clients @ {dt} with target")
    smpl.append(tmp)

te_client_groups = te_tgt.groupby("client_id").mon.max().reset_index()
te_client_groups = te_client_groups.groupby("mon").client_id.agg(list).to_dict()
for dt in ["2022-10-31", "2022-11-30", "2022-12-31"]:
    tmp = prepare_sample(
        dt,
        te_trx[(te_trx.event_date <= dt) & (te_trx.client_id.isin(te_client_groups[dt]))].copy(),
        te_tgt[te_tgt.mon <= dt],
        geo,
    )
    logging.info(f"added {tmp.client_id.nunique()} test clients @ {dt} for inference")
    smpl.append(tmp)

del tr_tgt, tr_trx, te_tgt, te_trx, tmp, geo
gc.collect()
smpl = pd.concat(smpl)
logging.info("feature engeneering")
smpl["id"] = smpl["client_id"] + "_" + smpl["dt"]
smpl["amount"] = pd.qcut(smpl["amount"], 128, duplicates="drop")
smpl["till_dt"] = (pd.to_datetime(smpl.dt) - smpl.event_time).dt.days.astype(int)
smpl["till_dt"] = pd.qcut(smpl["till_dt"], 128, duplicates="drop")
smpl["weekday"] = smpl.event_time.dt.weekday
smpl["hour"] = smpl.event_time.dt.hour
smpl.drop(columns=["client_id", "event_date", "mon"], inplace=True)


logging.info("preprocessing with ptls")
preprocessor = PandasDataPreprocessor(
    col_id="id",
    col_event_time="event_time",
    event_time_transformation="dt_to_timestamp",
    cols_category=cols_emb_dim.keys(),
    cols_first_item="target",
    return_records=False,
)
smpl = preprocessor.fit_transform(smpl)
smpl.loc[smpl.target.notnull(), "target"] = smpl.loc[smpl.target.notnull(), "target"].apply(
    lambda x: torch.tensor(x, dtype=torch.float32)
)
print(smpl.iloc[17])


# 3/16 clients for validation
dataset_train = MemoryMapDataset(
    smpl[(smpl.target.notnull()) & (~smpl.id.str[0].isin(["0", "1", "2", "3"]))].to_dict(orient="records"),
    i_filters=[ISeqLenLimit(max_seq_len=4096)],
)
dataset_valid = MemoryMapDataset(
    smpl[(smpl.target.notnull()) & (smpl.id.str[0].isin(["0", "1", "2", "3"]))].to_dict(orient="records"),
    i_filters=[ISeqLenLimit(max_seq_len=4096)],
)
dataset_test = MemoryMapDataset(
    smpl[smpl.target.isnull()].to_dict(orient="records"), i_filters=[ISeqLenLimit(max_seq_len=4096)]
)

datamodule = PtlsDataModule(
    train_data=SeqToTargetDataset(dataset_train, target_col_name="target"),
    valid_data=SeqToTargetDataset(dataset_valid, target_col_name="target"),
    test_data=SeqToTargetDataset(dataset_test, target_col_name="target"),
    train_batch_size=32,
    valid_batch_size=128,
    train_num_workers=16,
)

seq_encoder = RnnSeqEncoder(
    trx_encoder=TrxEncoder(
        embeddings_noise=0.003,
        embeddings={
            i: {"in": preprocessor.get_category_dictionary_sizes()[i], "out": j} for i, j in cols_emb_dim.items()
        },
    ),
    hidden_size=200,
    type="gru",
)

model = SequenceToTarget(
    seq_encoder=seq_encoder,
    head=torch.nn.Linear(200, 4),
    loss=torch.nn.BCEWithLogitsLoss(),
    metric_list=[MultilabelAUROC(num_labels=4, validate_args=False)],
    optimizer_partial=partial(torch.optim.Adam, lr=0.001),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=4, gamma=0.5),
    train_update_n_steps=999999999,
)


for ep in range(5):
    trainer = pl.Trainer(max_epochs=1, num_sanity_val_steps=0)
    trainer.fit(model, datamodule)
    preds = trainer.predict(model, dataloaders=datamodule.val_dataloader())
    pd.concat(preds).to_parquet("val_preds%d.pq" % ep)
    preds = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    pd.concat(preds).to_parquet("test_preds%d.pq" % ep)
