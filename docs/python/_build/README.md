# Building the Python method documents

Sources of the method documents in `docs/python/tools/` (`weights_of_evidence.docx`, `logistic_regression.docx`,
`random_forest.docx`, `svm.docx`, `xgboost.docx`, `rvm.docx`, `mlp.docx`, `elm.docx`, `cnn.docx`, `bayesian_network.docx`, `fuzzy_logic.docx`,
`boolean_logic.docx`, `logistic_geometric.docx`).

| File | Role |
|---|---|
| `experiments.py` | Runs every method on both demo projects (candidate search, robustness, reference run, sensitivity); writes `results/<method>_<dataset>.json` and `figures/*.png` |
| `univariate.py` | Single-layer AUC screening of all demo rasters → `results/univariate_*.json` |
| `content/<method>.js` | Text, equations and tables of one document (reads the JSON results) |
| `common.js` | Shared demo-data / protocol chapters and result tables |
| `lib.js` | Block → docx conversion (headings, equations as Word math, tables, figures, references) |
| `refs.js` | Reference list (`R('key', ...)`) |
| `build.js` | Writes the `.docx` files to `docs/python/tools/` |

## Rebuild

```bat
rem 1. results (≈ 1 h for all methods; --quick for a smoke run)
.venv\Scripts\python.exe docs\python\_build\univariate.py
.venv\Scripts\python.exe docs\python\_build\experiments.py
rem    one method / dataset:  --method random_forest --dataset carlin

rem 2. documents (Node.js with the npm package docx v9)
cd docs\python\_build
npm install docx@9          (once; or set NODE_PATH to a folder that has it)
node build.js               (all documents)
node build.js random_forest (one document)
```

Text in `content/*.js` uses a small markup: `**bold**`, `*italic*`, `` `code` ``, inline math `$...$`, display
equations `['eq', 'latex-like', 'label']` referenced as `[[label]]`. Numbers quoted in the text are read from the JSON
results, so re-running the experiments updates the documents.
