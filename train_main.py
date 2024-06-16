import pandas as pd
import numpy as np
import datetime
import logging
from sklearn.metrics import roc_auc_score
import gc
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def end_of_next_month(x):
    res = datetime.datetime.strptime(x, "%Y-%m-%d")
    res = datetime.datetime(res.year + int(res.month / 12), ((res.month % 12) + 1), 1)
    res = datetime.datetime(res.year + int(res.month / 12), ((res.month % 12) + 1), 1)
    res = res - datetime.timedelta(days=1)
    return res.strftime("%Y-%m-%d")


def get_hist_target_vars(tgt):
    tgt["target_any"] = tgt[["target_1", "target_2", "target_3", "target_4"]].max(1)
    tgt["since"] = pd.to_datetime(tgt.mon).dt.month.max() - pd.to_datetime(tgt.mon).dt.month
    res = tgt.groupby("client_id")[["target_1", "target_2", "target_3", "target_4", "target_any"]].mean()
    res.columns = ["hist_%s_mean" % i for i in res.columns]
    res["hist_target_any_sum"] = (
        tgt.groupby("client_id")[["target_1", "target_2", "target_3", "target_4"]].sum().sum(1)
    )
    for i in ["target_1", "target_2", "target_3", "target_4", "target_any"]:
        res["hist_%s_since_last" % i] = tgt[tgt[i] == 1].groupby("client_id").since.min()
        res["hist_%s_since_first" % i] = tgt[tgt[i] == 1].groupby("client_id").since.max()
    res["hist_target_nunique"] = (res[["hist_target_%d_mean" % i for i in range(1, 5)]] > 0).sum(1)
    return res


def get_trx_vars(trx):
    # nuniques for each field. trn count last m and ratios
    res = trx.groupby("client_id").event_time.agg(["count", "max"])
    res.columns = ["trx_cnt", "trx_since_last"]
    res["trx_nunique_dt"] = trx.groupby("client_id").event_date.nunique()
    res["trx_cnt_l1m"] = trx[
        (trx.event_time.max() - trx.event_time).dt.days.astype(int) <= 30
    ].client_id.value_counts()
    res["trx_sh_l1m"] = res["trx_cnt_l1m"] / res["trx_cnt"]
    res["trx_cnt_l3m"] = trx[
        (trx.event_time.max() - trx.event_time).dt.days.astype(int) <= 90
    ].client_id.value_counts()
    res["trx_sh_l3m"] = res["trx_cnt_l3m"] / res["trx_cnt"]

    res["trx_amount_mean"] = trx.groupby("client_id").amount.mean()

    for i in [54, 38, 37, 51, 25, 40, 56, 41, 16, 52]:
        res["trx_%d_mean" % i] = trx[trx.event_type == i].groupby("client_id").amount.mean()
        res["trx_%d_min" % i] = trx[trx.event_type == i].groupby("client_id").amount.min()
        res["trx_%d_sum" % i] = trx[trx.event_type == i].groupby("client_id").amount.sum()
        res["trx_%d_max" % i] = trx[trx.event_type == i].groupby("client_id").amount.max()
        res["trx_%d_count" % i] = trx[trx.event_type == i].groupby("client_id").amount.count()
        res["trx_%d_sh" % i] = res["trx_%d_count" % i] / res["trx_cnt"]
    #     tmp=trx.groupby('client_id')[list(trx.columns.difference(['client_id','event_time','amount','event_date']))].nunique()
    #     tmp.columns = ['trx_%s_nunique'%i for i in tmp.columns]
    #     res=res.merge(tmp,left_index=True,right_index=True)
    return res


def get_dial_vars(dials):
    res = dials.groupby("client_id")[["dial_pred_1", "dial_pred_2", "dial_pred_3", "dial_pred_4"]].mean()
    res.columns = [i + "_mean" for i in res.columns]
    tmp = dials.groupby("client_id")[["dial_pred_1", "dial_pred_2", "dial_pred_3", "dial_pred_4"]].max()
    tmp.columns = [i + "_max" for i in tmp.columns]
    res = res.merge(tmp, left_index=True, right_index=True)
    res["dial_cnt"] = dials.groupby("client_id").event_time.count()
    res["dial_since_last"] = dials.groupby("client_id").event_time.max()
    return res


def get_geo_vars(geo):
    res = geo.groupby("client_id")[["event_date", "geohash_4"]].nunique()
    res.columns = ["geo_nunique_dt", "geo_nunique"]
    res["geo_cnt"] = geo.groupby("client_id").cnt.sum()
    top = geo.groupby(["client_id", "geohash_4"]).cnt.sum().reset_index()
    top = top.sort_values(["client_id", "cnt"], ascending=False)
    top["rn"] = top.groupby("client_id").geohash_4.cumcount()
    top = top[top.rn == 0]
    res["geo_top"] = pd.Series(index=top.client_id.values, data=top.geohash_4.values)
    res["geo_since_last"] = pd.to_datetime(geo.groupby("client_id").event_date.max())
    return res


def prepare_sample(dt, target_hist, trx, dials, geo, target=None):
    tmp = get_hist_target_vars(target_hist).reset_index()
    if target is not None:
        tmp = tmp.merge(target, on="client_id")
        tmp.drop(columns=["mon"], inplace=True)
    tmp["dt"] = dt
    tmp = tmp.merge(get_trx_vars(trx).reset_index(), on="client_id", how="left")
    tmp["trx_since_last"] = (pd.to_datetime(tmp.dt) - tmp["trx_since_last"]).dt.days.astype(float)
    tmp = tmp.merge(get_dial_vars(dials).reset_index(), on="client_id", how="left")
    tmp["dial_since_last"] = (pd.to_datetime(tmp.dt) - tmp["dial_since_last"]).dt.days.astype(float)
    tmp = tmp.merge(get_geo_vars(geo).reset_index(), on="client_id", how="left")
    tmp["geo_since_last"] = (pd.to_datetime(tmp.dt) - tmp["geo_since_last"]).dt.days.astype(float)
    return tmp


logging.info("loading data")
tr_tgt = pd.read_parquet("data/train_target.parquet")
tr_tgt = tr_tgt[tr_tgt.client_id.str[0].isin(["0", "1", "2", "3"])]
te_tgt = pd.read_parquet("data/test_target_b.parquet")
te_client_groups = te_tgt.groupby("client_id").mon.max().reset_index()
te_client_groups = te_client_groups.groupby("mon").client_id.agg(list).to_dict()

tr_trx = pd.read_parquet("data/trx_train.parquet")
tr_trx = tr_trx[tr_trx.client_id.isin(tr_tgt.client_id)]
te_trx = pd.read_parquet("data/trx_test.parquet")
te_trx = te_trx[te_trx.client_id.isin(te_tgt.client_id)]
tr_trx["event_date"] = tr_trx.event_time.dt.date.astype(str)
te_trx["event_date"] = te_trx.event_time.dt.date.astype(str)

dial = pd.read_parquet("dial_preds.pq")
dial = dial[(dial.client_id.isin(tr_tgt.client_id)) | (dial.client_id.isin(te_tgt.client_id))]
dial["event_date"] = dial.event_time.dt.date.astype(str)

tr_geo = pd.read_parquet("data/geo_train.parquet")
tr_geo = tr_geo[tr_geo.client_id.isin(tr_tgt.client_id)]
te_geo = pd.read_parquet("data/geo_test.parquet")
te_geo = te_geo[te_geo.client_id.isin(te_tgt.client_id)]
tr_geo["event_date"] = tr_geo.event_time.dt.date.astype(str)
tr_geo = tr_geo.groupby(["client_id", "event_date", "geohash_4"]).geohash_5.count().reset_index()
tr_geo.columns = ["client_id", "event_date", "geohash_4", "cnt"]
te_geo["event_date"] = te_geo.event_time.dt.date.astype(str)
te_geo = te_geo.groupby(["client_id", "event_date", "geohash_4"]).geohash_5.count().reset_index()
te_geo.columns = ["client_id", "event_date", "geohash_4", "cnt"]

smpl = []
for dt in ["2022-10-31", "2022-11-30", "2022-12-31"]:
    tmp = prepare_sample(
        dt,
        target_hist=tr_tgt[tr_tgt.mon <= dt].copy(),
        trx=tr_trx[tr_trx.event_date <= dt],
        dials=dial[dial.event_date <= dt],
        geo=tr_geo[tr_geo.event_date <= dt],
        target=tr_tgt[tr_tgt.mon == end_of_next_month(dt)],
    )
    smpl.append(tmp)
    logging.info(f"added {tmp.client_id.nunique()} train clients @ {dt} with target")
for dt in ["2022-10-31", "2022-11-30"]:
    tmp = prepare_sample(
        dt,
        target_hist=te_tgt[(te_tgt.mon <= dt) & (te_tgt.client_id.str[0].isin(["0", "1", "2", "3"]))].copy(),
        trx=te_trx[te_trx.event_date <= dt],
        dials=dial[dial.event_date <= dt],
        geo=te_geo[te_geo.event_date <= dt],
        target=te_tgt[(te_tgt.mon == end_of_next_month(dt)) & (te_tgt.client_id.str[0].isin(["0", "1", "2", "3"]))],
    )
    smpl.append(tmp)
    logging.info(f"added {tmp.client_id.nunique()} test clients @ {dt} with target")
for dt in ["2022-10-31", "2022-11-30", "2022-12-31"]:
    tmp = prepare_sample(
        dt,
        target_hist=te_tgt[(te_tgt.mon <= dt) & (te_tgt.client_id.isin(te_client_groups[dt]))].copy(),
        trx=te_trx[te_trx.event_date <= dt],
        dials=dial[dial.event_date <= dt],
        geo=te_geo[te_geo.event_date <= dt],
        target=None,
    )
    smpl.append(tmp)
    logging.info(f"added {tmp.client_id.nunique()} test clients @ {dt} for inference")

smpl = pd.concat(smpl)

smpl["id"] = smpl.client_id + "_" + smpl.dt
smpl["dt"] = smpl.dt.str[5:7].astype(int)
top = smpl.geo_top.value_counts().head(30).index.values
smpl.loc[~smpl.geo_top.isin(top), "geo_top"] = -1
smpl["geo_top"] = pd.Categorical(smpl.geo_top.astype(str))

nn_preds = None
for ep in range(5):
    tmp = pd.concat(
        [
            pd.read_parquet("test_preds%d.pq" % ep),
            pd.read_parquet("val_preds%d.pq" % ep),
        ]
    )[
        ["id", "out_0000", "out_0001", "out_0002", "out_0003"]
    ].set_index("id")
    if nn_preds is None:
        nn_preds = tmp
    else:
        nn_preds += tmp
nn_preds.columns = ["nn1", "nn2", "nn3", "nn4"]
smpl = smpl.merge(nn_preds, left_on="id", right_index=True, how="left")


logging.info("crossvalidation")
cols = list(smpl.columns.difference(["client_id", "id", "target_1", "target_2", "target_3", "target_4"]))
aucs = {}
models = {
    1: LGBMClassifier(n_estimators=400, max_depth=2, learning_rate=0.03, random_state=1, verbose=-1),
    2: LGBMClassifier(n_estimators=500, max_depth=2, learning_rate=0.02, random_state=1, verbose=-1),
    3: LGBMClassifier(n_estimators=700, max_depth=2, learning_rate=0.03, random_state=1, verbose=-1),
    4: LGBMClassifier(n_estimators=400, max_depth=2, learning_rate=0.03, random_state=1, verbose=-1),
}
catboosts = {
    1: CatBoostClassifier(eval_metric="AUC", learning_rate=0.03, depth=6, iterations=1000, verbose=0),
    2: CatBoostClassifier(eval_metric="AUC", learning_rate=0.03, depth=6, iterations=1000, verbose=0),
    3: CatBoostClassifier(eval_metric="AUC", learning_rate=0.03, depth=6, iterations=1000, verbose=0),
    4: CatBoostClassifier(eval_metric="AUC", learning_rate=0.03, depth=6, iterations=1000, verbose=0),
}
for fold in range(4):
    logging.info("train fold %d" % fold)
    aucs[fold] = {}
    tr_idx = (smpl.client_id.str[0] != str(fold)) & (smpl.target_1.notnull())
    val_idx = (smpl.client_id.str[0] == str(fold)) & (smpl.target_1.notnull())
    for target in range(1, 5):
        models[target].fit(smpl[tr_idx][cols], smpl[tr_idx]["target_%d" % target])
        pred0 = models[target].predict_proba(smpl[val_idx][cols])[:, 1]
        catboosts[target].fit(smpl[tr_idx][cols], smpl[tr_idx]["target_%d" % target], cat_features=["geo_top"])
        pred1 = catboosts[target].predict_proba(smpl[val_idx][cols])[:, 1]
        pred = (pred0 + pred1) / 2.0
        auc = roc_auc_score(smpl[val_idx]["target_%d" % target], pred)
        aucs[fold][target] = auc
aucs = pd.DataFrame(aucs).T
print(aucs)
print(aucs.mean())
print(aucs.mean().mean())

logging.info("final fit")
tr_idx = smpl.target_1.notnull()
subm_idx = smpl.target_1.isnull()
for target in range(1, 5):
    models[target].fit(smpl[tr_idx][cols], smpl[tr_idx]["target_%d" % target])
    pred0 = models[target].predict_proba(smpl[subm_idx][cols])[:, 1]
    catboosts[target].fit(smpl[tr_idx][cols], smpl[tr_idx]["target_%d" % target], cat_features=["geo_top"])
    pred1 = catboosts[target].predict_proba(smpl[subm_idx][cols])[:, 1]
    smpl.loc[subm_idx, "target_%d" % target] = (pred0 + pred1) / 2.0
smpl[subm_idx][["client_id", "target_1", "target_2", "target_3", "target_4"]].to_csv("submission.csv")
