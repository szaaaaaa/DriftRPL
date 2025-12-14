import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from driftrpl.datasets.gas_drift import load_gas_drift_stream

def main():
    s = load_gas_drift_stream()
    print("OK")
    print("X:", s.X.shape, "y:", s.y.shape)
    print("batch unique:", len(set(s.batch.tolist())))
    print("gas_id unique:", len(set(s.gas_id.tolist())))
    print("y range:", float(s.y.min()), float(s.y.max()))

if __name__ == "__main__":
    main()
