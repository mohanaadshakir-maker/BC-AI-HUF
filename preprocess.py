"""
NSL-KDD preprocessing pipeline for the BC-AI-HUF AI layer rebuild.

Data source: NSL-KDD (Tavallaee et al., 2009), an improved, de-duplicated
successor to KDD Cup 1999, widely used as an intrusion-detection benchmark.
Files downloaded from a standard public mirror of the official UNB release:
  KDDTrain+.txt (125,973 rows), KDDTest+.txt (22,544 rows)

Key property exploited later for the zero-day test: KDDTest+ contains 17
attack subtypes that never appear in KDDTrain+ (apache2, httptunnel,
mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack,
snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm). This is a
genuine, data-driven analogue of "unseen attack" / zero-day generalisation,
verified empirically (see explore.py output), not asserted from memory.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json
import os

COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
"root_shell","su_attempted","num_root","num_file_creations","num_shells",
"num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
"srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
"same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
"dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
"label","difficulty"]

CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLS if c not in CATEGORICAL + ["label", "difficulty"]]

# Standard NSL-KDD attack-category mapping (DoS / Probe / R2L / U2R), used
# throughout the NSL-KDD literature (Tavallaee et al. 2009 and successors).
ATTACK_CATEGORY = {
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "worm": "DoS", "mailbomb": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
    "normal": "normal",
}


def load_raw(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "KDDTrain+.txt"), names=COLS)
    test = pd.read_csv(os.path.join(data_dir, "KDDTest+.txt"), names=COLS)
    return train, test


def build_features(train, test):
    # One-hot encode categorical columns; fit ONLY on train, then align test
    # columns to train's, filling unseen categories with 0 (standard practice
    # to avoid leaking test-only categorical levels into the feature space).
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(train[CATEGORICAL])
    train_cat = pd.DataFrame(enc.transform(train[CATEGORICAL]),
                              columns=enc.get_feature_names_out(CATEGORICAL),
                              index=train.index)
    test_cat = pd.DataFrame(enc.transform(test[CATEGORICAL]),
                             columns=enc.get_feature_names_out(CATEGORICAL),
                             index=test.index)

    scaler = StandardScaler()
    train_num = pd.DataFrame(scaler.fit_transform(train[NUMERIC]), columns=NUMERIC, index=train.index)
    test_num = pd.DataFrame(scaler.transform(test[NUMERIC]), columns=NUMERIC, index=test.index)

    X_train = pd.concat([train_num, train_cat], axis=1)
    X_test = pd.concat([test_num, test_cat], axis=1)

    y_train_bin = (train["label"] != "normal").astype(int)
    y_test_bin = (test["label"] != "normal").astype(int)

    y_train_cat = train["label"].map(ATTACK_CATEGORY)
    y_test_cat = test["label"].map(ATTACK_CATEGORY)

    novel_labels = sorted(set(test["label"].unique()) - set(train["label"].unique()))
    is_novel_test = test["label"].isin(novel_labels)

    meta = {
        "n_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "novel_attack_labels": novel_labels,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_novel_in_test": int(is_novel_test.sum()),
    }
    return X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat, test["label"], is_novel_test, meta


if __name__ == "__main__":
    data_dir = "/home/claude/bc_ai_huf_simulation/data"
    out_dir = "/home/claude/bc_ai_huf_simulation/data/processed"
    os.makedirs(out_dir, exist_ok=True)

    train, test = load_raw(data_dir)
    X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat, test_label_raw, is_novel_test, meta = build_features(train, test)

    X_train.to_parquet(os.path.join(out_dir, "X_train.parquet"))
    X_test.to_parquet(os.path.join(out_dir, "X_test.parquet"))
    y_train_bin.to_frame("y").to_parquet(os.path.join(out_dir, "y_train_bin.parquet"))
    y_test_bin.to_frame("y").to_parquet(os.path.join(out_dir, "y_test_bin.parquet"))
    y_train_cat.to_frame("y").to_parquet(os.path.join(out_dir, "y_train_cat.parquet"))
    y_test_cat.to_frame("y").to_parquet(os.path.join(out_dir, "y_test_cat.parquet"))
    test_label_raw.to_frame("label").to_parquet(os.path.join(out_dir, "test_label_raw.parquet"))
    is_novel_test.to_frame("is_novel").to_parquet(os.path.join(out_dir, "is_novel_test.parquet"))

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
