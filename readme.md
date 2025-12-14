# DriftRPL

DriftRPL is a lightweight and reproducible experimental framework for online time-series prediction under concept drift, featuring a replay-based learning mechanism and systematic experiment logging.

This repository is structured to support paper-level reproducibility, including deterministic seeds, explicit configuration via command-line arguments, and traceable experiment outputs.

---

## 1. Environment

### 1.1 Requirements

- Python 3.10 or later (3.10 recommended)
- Windows, Linux, or macOS
- GPU is optional (CUDA-supported GPU recommended for full experiments)

### 1.2 Installation

It is recommended to use a virtual environment.

**Create a virtual environment:**

```bash
python -m venv .venv
```

**Activate the environment:**

* **Windows:**
    ```cmd
    .venv\Scripts\activate
    ```
* **Linux / macOS:**
    ```bash
    source .venv/bin/activate
    ```

**Upgrade pip and install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Running the Code

All experiments are executed through a single entry script:

```bash
python driftrpl_all_things_here.py
```

To inspect all supported command-line arguments:

```bash
python driftrpl_all_things_here.py -h
```

---

## 3. Smoke Test (Sanity Check)

A smoke test is recommended to verify that:

* The code runs end-to-end
* Outputs are generated correctly
* Logs and metrics are written successfully

**Example lightweight command:**

```bash
python driftrpl_all_things_here.py \
  --device cpu \
  --out_dir outputs \
  --seeds 0 \
  --total_len 2000 \
  --segment_len 200 \
  --steps_K 1 \
  --batch_B 8
```

**Expected behavior:**

* A new experiment directory is created under `outputs/`
* Log files and metric files appear in that directory
* Runtime is short (typically seconds to a few minutes)

If some arguments above are not supported, use `-h` to identify the closest equivalents. The purpose of this run is to validate the execution pipeline with minimal data and computation.

---

## 4. Full Experiment Run

A full experiment typically includes:

* Multiple random seeds
* Full sequence length
* Standard training steps
* One or more concept drift types

**Example command:**

```bash
python driftrpl_all_things_here.py \
  --device cuda \
  --out_dir outputs \
  --seeds 0 1 2 3 4
```

Additional arguments may be available for:

* Model selection
* Drift type specification
* Replay buffer size
* Optimization and training hyperparameters

Use the `-h` flag to inspect all supported options.

---

## 5. Outputs

All generated artifacts are stored in the `outputs/` directory. Each experiment run creates a dedicated subdirectory that may contain:

* A snapshot of the full configuration (command-line arguments)
* Concise training logs
* Loss values written incrementally to CSV files
* Final and intermediate evaluation metrics
* Model checkpoints (if enabled)
* Figures or plots (if enabled)

Each experiment directory uniquely identifies a run and enables full traceability from results back to configuration.

---

## 6. Reproducibility Guidelines

To reproduce identical results:

1.  Fix all random seeds.
2.  Use identical hyperparameters.
3.  Use the same software environment.
4.  Keep device settings consistent (CPU or GPU).

**Recommended protocol for paper results:**

1.  First run a smoke test on CPU.
2.  Then run full experiments on GPU.
3.  Aggregate metrics from the final output files in `outputs/`.

---

## 7. Repository Structure

```text
.
├── driftrpl_all_things_here.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── outputs/
```

---

## 8. Troubleshooting

### 8.1 No outputs are generated
* Ensure the `outputs/` directory is writable.
* Explicitly pass ` --out_dir outputs`.
* Check console logs for the experiment directory path.

### 8.2 GPU is not used
* Verify CUDA availability in your environment.
* Ensure a CUDA-enabled PyTorch build is installed.
* Use the correct device argument (for example, `--device cuda`).

### 8.3 Training is slow
* Reduce sequence length or number of training steps.
* Use fewer random seeds.
* Switch to a simpler model if supported.
* Run on GPU when available.

---

## 9. Citation

If you use this code in academic research, please cite the associated paper (to be added).

## 10. License

This project is released under the MIT License.  
See the `LICENSE` file for details.
