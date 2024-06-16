import pandas as pd
import numpy as np
import datetime
from lightgbm import LGBMClassifier


def end_of_previous_month(x):
    first = x.replace(day=1)
    return first - datetime.timedelta(days=1)


tgt = pd.concat([pd.read_parquet("data/train_target.parquet"), pd.read_parquet("data/test_target_b.parquet")])
tgt["mon"] = pd.to_datetime(tgt["mon"]).apply(end_of_previous_month).dt.month

dial = pd.concat([pd.read_parquet("data/dial_test.parquet"), pd.read_parquet("data/dial_train.parquet")])
dial["mon"] = dial.event_time.dt.month
dial = dial.merge(tgt, on=["client_id", "mon"], how="left")

emb = np.stack(dial.embedding.values)
dial.drop(columns=["embedding"], inplace=True)

tr_idx = (dial.target_1.notnull()) & (~dial.client_id.str[0].isin(["0", "1", "2", "3"]))
for i in range(1, 5):
    lgb = LGBMClassifier(n_estimators=500, max_depth=2, verbose=-1, random_state=1)
    lgb.fit(emb[tr_idx], dial["target_%d" % i][tr_idx])
    dial["dial_pred_%d" % i] = lgb.predict_proba(emb, raw_score=True)
dial[["client_id", "event_time", "dial_pred_1", "dial_pred_2", "dial_pred_3", "dial_pred_4"]].to_parquet(
    "dial_preds.pq"
)
