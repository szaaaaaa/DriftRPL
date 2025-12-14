from pathlib import Path

RAW_DIR = Path("data/gas_drift/dat_type")   # 你的 batch*.dat 所在目录  

def main():
    files = sorted(RAW_DIR.glob("batch*.dat"))
    if not files:
        print("No batch*.dat found")
        return

    f = files[0]
    print("Inspecting file:", f)
    print("-" * 60)

    with f.open("r", encoding="utf-8", errors="ignore") as fh:
        for i in range(10):
            line = fh.readline()
            if not line:
                break
            print(f"Line {i}: {repr(line)}")

if __name__ == "__main__":
    main()
