# Python environment

## Runtime

- Python 3.12.14, 64-bit.
- Windows 64-bit is the reference platform.
- Node.js 24.19.0 and pdfjs-dist 5.6.205 are used by the PDF coordinate-extraction stage.
- Exact Python package versions are listed in `PACKAGE_VERSIONS.md` and the requirements files.

## Install

Run the following commands from the repository root. If several Python versions are installed, select the Python 3.12.14 executable explicitly.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment/requirements-preprocessing.txt
.\.venv\Scripts\python.exe code/verify.py
```

The interpreter path in the remaining examples is abbreviated to `python`. Use `.\.venv\Scripts\python.exe` on Windows, or the active environment's interpreter. Do not install dependencies into the system Python.

The notebook runner uses the Python standard library and does not require Jupyter. A notebook editor is optional; select this virtual environment as its kernel.

## Main figures

```text
python code/notebook_runtime.py code/Main_Figures.ipynb
```

This command reads the included plotting tables. It does not estimate the underlying model or regenerate bootstrap inputs.

## External model inputs

Obtain the files listed in `input/required_files.json` from their cited providers or an authorized data deposit. Preserve the listed relative paths beneath `DATA_ROOT`. Choose a short, new `RUN_ROOT` outside this repository; long nested Windows paths can exceed filesystem limits.

```text
python code/prepare.py --source-root DATA_ROOT --profile full --check-only
python code/prepare.py --source-root DATA_ROOT --destination RUN_ROOT --profile full
```

The `models` profile selects numerical inputs; `figures` selects supplementary plotting inputs; `full` includes both. Input hashes must match the manifest. Expected result files are validation targets, not substitutes for calculation.

## Model stages

Run the stages in order:

```text
python code/run.py raw --workspace RUN_ROOT
npm install --prefix environment
python code/run.py seed-coordinates --workspace RUN_ROOT --pdfjs environment/node_modules/pdfjs-dist/legacy/build/pdf.mjs
python code/run.py seed --workspace RUN_ROOT
python code/run.py baci --workspace RUN_ROOT
python code/run.py historical --workspace RUN_ROOT
python code/run.py allocation --workspace RUN_ROOT
python code/run.py routes --workspace RUN_ROOT
```

Conditional uncertainty and structural comparisons:

```text
python code/run.py corridor-uncertainty --workspace RUN_ROOT
python code/run.py raw-uncertainty --workspace RUN_ROOT
python code/run.py allocation-uncertainty --workspace RUN_ROOT
python code/run.py no-propagation --workspace RUN_ROOT
python code/run.py route-avoidance --workspace RUN_ROOT
python code/run.py preferences --workspace RUN_ROOT
python code/run.py figures --workspace RUN_ROOT
```

Use `--dry-run` to inspect commands without executing them. Computational cost depends on hardware; the allocation uncertainty stage comprises 20 paired high-precision jobs. Do not overwrite completed jobs or weaken numerical acceptance tolerances.

`requirements-solver-windows.lock` records Windows wheel hashes; it is not a cross-platform lock. A complete clean-install and cross-platform validation is not established.
