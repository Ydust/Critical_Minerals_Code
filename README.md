# Critical-mineral supply diversification

## Files

- `code/`: model calculations, plotting notebooks and execution utilities.
- `input/`: model input index, source references, parameters and map assets.
- `output/`: main-figure plotting tables.
- `environment/`: Python setup, dependency versions and configuration files.

## Run

Follow [environment/SETUP.md](environment/SETUP.md) to configure Python. Package versions are listed in [environment/PACKAGE_VERSIONS.md](environment/PACKAGE_VERSIONS.md).

Run `code/Main_Figures.ipynb` to reproduce Figures 1–5 from the included plotting data. The notebook can also be executed without a notebook editor:

```text
python code/notebook_runtime.py code/Main_Figures.ipynb
```

Figures are written to `output/figures/`. The Excel workbook contains one sheet per main-figure subpanel; CSV equivalents are in `output/main_figures/`.

Complete model calculations require the external files indexed in `input/required_files.json`. These raw inputs are not included. Data access and redistribution remain subject to the original providers' terms.
