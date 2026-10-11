"""Build and execute all notebooks against the real data. Usage: python scripts/build_notebooks.py [01 02 ...]"""
import sys, pathlib
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nb_defs_a import NB01, NB02
from nb_defs_b import NB03, NB04, NB05

NBS = {"01": ("01_preregistration_pipeline_check", NB01), "02": ("02_statistical_tests_anova", NB02),
       "03": ("03_pca_clustering", NB03), "04": ("04_timeseries_forecasting", NB04), "05": ("05_capstone_predictive_modelling", NB05)}
OUT = HERE.parent / "notebooks"

def build(key):
    name, cells = NBS[key]
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT)}})
    nbf.write(nb, OUT / f"{name}.ipynb")
    print("built", name)

if __name__ == "__main__":
    for k in (sys.argv[1:] or NBS): build(k)
