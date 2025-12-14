from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/gas_drift/dat_type")  # 你现在文件就在这里
OUT_DIR = Path("data/gas_drift/csv_type")  # 输出目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FEATS = 128  # 你文件里是 1:... 到 128:...

def parse_line(line: str):
    line = line.strip()
    if not line:
        return None


    # 1) 拆出 "gas_id;concentration" 和 "index:value ..."
    try:
        head, tail = line.split(" ", 1)
    except ValueError:
        # 这一行没有特征部分，跳过
        return None

    if ";" not in head:
        return None

    gas_str, conc_str = head.split(";", 1)
    gas_id = int(float(gas_str))
    y = float(conc_str)

    x = np.zeros(N_FEATS, dtype=np.float32)

    # 2) 解析 "k:v" 对
    for token in tail.split():
        if ":" not in token:
            continue
        k_str, v_str = token.split(":", 1)
        k = int(k_str)
        if 1 <= k <= N_FEATS:
            x[k - 1] = float(v_str)

    return gas_id, y, x

def load_batch(path: Path, batch_idx: int):
    gas_ids, ys, xs, batches = [], [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            out = parse_line(line)
            if out is None:
                continue
            gas_id, y, x = out
            gas_ids.append(gas_id)
            ys.append(y)
            xs.append(x)
            batches.append(batch_idx)

    if not xs:
        return None

    X = np.vstack(xs)
    y = np.array(ys, dtype=np.float32)
    gas_ids = np.array(gas_ids, dtype=np.int32)
    batches = np.array(batches, dtype=np.int32)
    return X, y, gas_ids, batches

def main():
    files = sorted(RAW_DIR.glob("batch*.dat"))
    if not files:
        raise FileNotFoundError(f"No batch*.dat found in {RAW_DIR.resolve()}")

    X_all, y_all, gas_all, batch_all = [], [], [], []

    for i, f in enumerate(files, start=1):
        out = load_batch(f, batch_idx=i)
        if out is None:
            print(f"{f.name}: EMPTY")
            continue
        X, y, gas_ids, batches = out
        print(f"{f.name}: X={X.shape}, y={y.shape}, gas={gas_ids.shape}")
        X_all.append(X)
        y_all.append(y)
        gas_all.append(gas_ids)
        batch_all.append(batches)

    if not X_all:
        raise RuntimeError("No valid rows parsed. Check file contents.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    gas_ids = np.concatenate(gas_all)
    batches = np.concatenate(batch_all)

    # 保存：X.csv（128维特征 + gas_id），y.csv（浓度y + gas_id + batch）
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(1, N_FEATS + 1)])
    X_df.insert(0, "gas_id", gas_ids)

    y_df = pd.DataFrame({"y": y, "gas_id": gas_ids, "batch": batches})

    X_path = OUT_DIR / "X.csv"
    y_path = OUT_DIR / "y.csv"
    X_df.to_csv(X_path, index=False)
    y_df.to_csv(y_path, index=False)

    print("\nSaved:")
    print(" ", X_path, X_df.shape)
    print(" ", y_path, y_df.shape)
    print("y meaning: concentration")
    print("gas_id meaning: class label before ';'")

if __name__ == "__main__":
    main()
