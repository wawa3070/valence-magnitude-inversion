"""Build docs/index.html — the project page + data explorers.

Embeds two datasets into `page_template.html`:

1. the committed loan SHAP instances (`data/loan_synthetic_shap_instances.csv`), and
2. per-alert TreeSHAP for the network alerts, regenerated here by rerunning the
   pipeline of Section 3 of `Inversion_Experiment.ipynb` verbatim (Random Forest +
   SMOTE at `random_state=42`, 10% stratified split, TreeSHAP on the anomaly class,
   top-5 drivers per alert), then scanning the seed-42 shuffled predicted anomalies
   until 50 traps are collected — the exact selection the experiment uses.

Usage (from the repo root):
    pip install numpy pandas scikit-learn shap imbalanced-learn
    python docs/build_page.py
"""
import json
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

NET_SEED, NET_TOP_K, NET_N_TRAPS = 42, 5, 50
DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
DATA = os.path.join(ROOT, "data")


# ---------------- the notebook's two orderings ----------------
def tone_ordering(features, shap_vals):
    """'Good news first': positives by value descending, then negatives by |value| ascending."""
    pos = sorted([i for i, s in enumerate(shap_vals) if s > 0], key=lambda i: -shap_vals[i])
    neg = sorted([i for i, s in enumerate(shap_vals) if s <= 0], key=lambda i: abs(shap_vals[i]))
    return [features[i] for i in pos + neg]


def is_trap(shap_vals):
    """True when tone order != magnitude order — only then is inversion observable."""
    feats = [f"f{i}" for i in range(len(shap_vals))]
    mag = sorted(feats, key=lambda f: -abs(shap_vals[feats.index(f)]))
    return tone_ordering(feats, list(shap_vals)) != mag


# ---------------- loan instances (straight from the committed CSV) ----------------
def build_loan():
    df = pd.read_csv(os.path.join(DATA, "loan_synthetic_shap_instances.csv"))
    out = []
    for iid, g in df.sort_values(["instance_id", "position"]).groupby("instance_id", sort=False):
        out.append({"instance_id": iid, "feature_set": str(g["feature_set"].iloc[0]),
                    "seed": int(g["seed"].iloc[0]), "features": list(g["feature"]),
                    "shap": [float(v) for v in g["shap_value"]],
                    "decision": str(g["decision"].iloc[0]), "is_trap": bool(g["is_trap"].iloc[0])})
    print(f"loan instances: {len(out)} | traps: {sum(i['is_trap'] for i in out)}")
    return out


# ---------------- network alerts (Section 3 of the notebook) ----------------
def build_network():
    net_raw = pd.read_csv(os.path.join(DATA, "network_traffic.csv"))
    d = net_raw.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d["hour_of_day"] = d["time"].dt.hour
    d["day_of_week"] = d["time"].dt.dayofweek
    d = d.drop(columns=["time"])
    if "label_f" in d.columns:
        d = d.drop(columns=["label_f"])
    y = d["label"].astype(int)
    X = d.drop(columns=["label"])
    names = list(X.columns)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.10, random_state=NET_SEED, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(scaler.transform(Xtr), columns=names, index=Xtr.index)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=names, index=Xte.index)
    Xtr_r, ytr_r = SMOTE(random_state=NET_SEED).fit_resample(Xtr_s, ytr)
    rf = RandomForestClassifier(random_state=NET_SEED, n_jobs=-1).fit(Xtr_r, ytr_r)
    pred = rf.predict(Xte_s)
    report = classification_report(yte, pred, output_dict=True)
    auc = float(roc_auc_score(yte, rf.predict_proba(Xte_s)[:, 1]))
    print(classification_report(yte, pred))
    print("AUC-ROC:", round(auc, 4))

    sv = shap.TreeExplainer(rf).shap_values(Xte_s)
    if isinstance(sv, list):
        sv_anom = np.asarray(sv[1])
    else:
        sv = np.asarray(sv)
        sv_anom = sv[:, :, 1] if sv.ndim == 3 else sv

    idx_anom = np.where(pred == 1)[0]
    rng = np.random.default_rng(NET_SEED)
    rng.shuffle(idx_anom)

    raw_te = net_raw.loc[Xte.index]
    alerts, n_traps = [], 0
    for i in idx_anom:
        row = sv_anom[i]
        order = np.argsort(-np.abs(row))[:NET_TOP_K]
        feats = [names[j] for j in order]
        vals = [round(float(row[j]), 4) for j in order]
        trap = bool(is_trap(vals))
        n_traps += trap
        src = raw_te.iloc[int(i)]
        alerts.append({
            "alert_id": f"alert_{len(alerts):03d}",
            "row_index": int(Xte.index[i]),
            "time": str(src["time"]),
            "true_label": int(yte.iloc[i]),
            "proba": round(float(rf.predict_proba(Xte_s.iloc[[i]])[0, 1]), 3),
            "features": feats, "shap": vals,
            "is_trap": trap, "selected": bool(trap and n_traps <= NET_N_TRAPS),
            "raw": {k: float(src[k]) for k in
                    ["source_port", "destination_port", "protocol", "duration",
                     "packet_count", "bytes_sent", "bytes_received", "bytes_per_packet"]},
        })
        if n_traps >= NET_N_TRAPS:
            break

    print(f"alerts scanned: {len(alerts)} | traps: {n_traps} | "
          f"natural trap rate: {100 * n_traps / len(alerts):.1f}%")
    return {"feature_names": names, "n_scanned": len(alerts), "n_traps": int(n_traps),
            "auc": round(auc, 4),
            "precision_anom": round(report["1"]["precision"], 3),
            "recall_anom": round(report["1"]["recall"], 3),
            "f1_anom": round(report["1"]["f1-score"], 3),
            "n_test": int(len(yte)), "n_flagged": int(pred.sum()), "alerts": alerts}



# ---------------- real narratives (July 26 network run; committed CSVs) ----------------
NARR_ORDER = ["SmolLM3-3B", "gemini-3.6-flash", "GLM-5.2", "Claude Haiku 4.5",
              "Kimi K3", "Qwen 3.7 Max", "GPT-5.6 Luna"]
NARR_DISPLAY = {"gemini-3.6-flash": "Gemini 3.6 Flash"}
NARR_CONDS = ["C0_baseline", "C1_ordering", "C2_urgent_ordering"]


def build_narratives():
    df = pd.concat([pd.read_csv(os.path.join(DATA, "net50_c0_narratives.csv")),
                    pd.read_csv(os.path.join(DATA, "net50_c12_narratives.csv"))], ignore_index=True)
    tob = lambda x: 1 if str(x) in ("True", "1", "1.0") else 0
    by_idx = {}
    for k in range(NET_N_TRAPS):
        rec = {}
        for c in NARR_CONDS:
            rows = []
            for mi, m in enumerate(NARR_ORDER):
                g = df[(df.model == m) & (df.condition == c) & (df.idx == k)]
                if len(g):
                    r = g.iloc[0]
                    rows.append([mi, tob(r.faithful), tob(r.tone_ordered), str(r.narrative)])
            if rows:
                rec[c] = rows
        by_idx[str(k)] = rec
    print(f"narratives: {sum(len(v.get(c, [])) for v in by_idx.values() for c in NARR_CONDS)} across {len(by_idx)} selected alerts")
    return {"models": [NARR_DISPLAY.get(m, m) for m in NARR_ORDER], "conds": NARR_CONDS, "byIdx": by_idx}


def main():
    loan = json.dumps(build_loan())
    net = json.dumps(build_network())
    narr = json.dumps(build_narratives(), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = open(os.path.join(DOCS, "page_template.html")).read()
    html = tpl.replace("__LOAN_JSON__", loan).replace("__NET_JSON__", net).replace("__NET_NARR_JSON__", narr)
    out = os.path.join(DOCS, "index.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out, f"({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
