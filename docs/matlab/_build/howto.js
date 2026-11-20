// "How to prepare data and test the method" section, shared by the method documents M1-M9.
// howto({ code, tasks: ['clf'] | ['clf','reg'], scripts: {clf:'run_x_classification.m', reg:...},
//         params: [lines of cfg.params examples], notes: [method-specific preparation bullets],
//         checks: [method-specific result checks], verify: [how to repeat the synthetic checks] })

function howto(o) {
  const clf = o.tasks.includes('clf'), reg = o.tasks.includes('reg');
  const B = [];
  B.push(['h1', '6 How to prepare data and test the method']);

  // ------------------------------------------------------------------ 6.1 data
  B.push(['h2', '6.1 Prepare the data']);
  B.push(['p', 'The MATLAB tool reads a pre-processed database; it does not need Python. The database can be produced by the Python pre-processing tool or by any GIS.']);
  B.push(['p', '**Evidence rasters.** One folder (the “processed” folder) containing:']);
  B.push(['bullets', [
    '`layers.csv` with the columns `name, file, group, kind, description` — one row per raster. `kind` is `continuous` or `categorical`. A layer with `group = mask` (by default named `study_mask`, 1 inside, 0 or nodata outside) defines the study area; layers with `group = reference` (e.g. an expert map) are never used as evidence.',
    '`rasters/<name>.tif`: one GeoTIFF per layer, **all on the same grid** (same CRS, cell size, extent and alignment) with nodata set. Resample, clip and snap the layers to one grid in the GIS before export.',
    'Optional `rasters/<name>.classes.csv` (`value, label, …`) for categorical layers, so that class names appear in the results.',
    'Categorical layers are one-hot encoded (one column per class). A classified layer whose class values are physical quantities (e.g. magnetic intensity classes in nT) can be used as a number with `cfg.features.as_numeric`. Missing values are imputed (median / all-zero one-hot) unless `cfg.features.missing = \'exclude\'`.',
  ]]);
  const pts = [];
  if (clf) pts.push('**Deposits** (classification): a CSV with columns `x, y` in the CRS of the rasters, one row per deposit or occurrence (QGIS: *Export → Save Features As → CSV, GEOMETRY = AS_XY*), or a point shapefile. Other columns are kept (e.g. a `RANDOM` column for a rule-based split). Deposits outside the study area are ignored; several deposits in one cell count once.');
  if (reg) pts.push('**Grade samples** (regression): a CSV or point shapefile with `x, y` and a numeric grade column (e.g. drill-hole composites, `Au_gpt`). For the default log10 model, grades ≤ 0 are dropped (`cfg.nonpositive = \'drop\'`) or set to half the minimum (`\'halfmin\'`); samples in the same cell are averaged.');
  B.push(['p', '**Points.**']);
  B.push(['bullets', pts]);
  B.push(['p', '**Choosing the evidence.**']);
  const ev = [
    'Select layers that represent the mineral system (source, pathway, trap, deposition) at a cell size at which deposits are separable (ideally one deposit per cell).',
    'Avoid leakage: do not use layers derived from the target points themselves, such as a distance-to-known-deposit map, or (for regression) geochemical surfaces interpolated from the same samples that are predicted (`cfg.exclude_layers`).',
  ];
  if (clf) ev.push('Background cells (negatives) are drawn automatically beyond `cfg.negatives.min_distance` (m) from every known deposit; set it to the footprint of a deposit plus its alteration halo.');
  B.push(['bullets', ev]);
  if (o.notes && o.notes.length) {
    B.push(['p', '**Specific to this method.**']);
    B.push(['bullets', o.notes]);
  }

  // ------------------------------------------------------------------ 6.2 run
  B.push(['h2', '6.2 Configure and run']);
  const steps = ['Start MATLAB R2024a (Statistics and Machine Learning, Mapping and Image Processing Toolboxes) and open the run script:' +
    (clf ? ` \`classification/${o.scripts.clf}\`` : '') + (clf && reg ? ' or' : '') + (reg ? ` \`regression/${o.scripts.reg}\`` : '') + '.',
    'For the demo data keep `dataset = \'carlin\'` (or `\'curnamona\'`). For your own data replace the line `cfg = …demo_config(…)` by your settings (example below).',
    'Optionally set method parameters (`cfg.params`) or a tuning grid (`cfg.tune.grid`); otherwise the defaults and the grid of Section 4 are used.',
    'Run the script (F5). The log is printed in the Command Window; at the end the results window opens and all outputs are written to `matlab/results/<dataset>/<task>_<method>_<date_time>/` (`result.mat`, GeoTIFF maps, CSV tables, `log.txt`).',
  ];
  B.push(['steps', steps]);
  const code = ['cfg = struct();',
    "cfg.data_dir   = 'D:\\project\\processed';           % folder with layers.csv + rasters/",
    clf ? "cfg.points_csv = 'D:\\project\\deposits.csv';        % x, y in the raster CRS" : "cfg.points_csv = 'D:\\project\\composites.csv';      % x, y + grade column",
    "cfg.layers     = {'geology', 'mag_rtp', 'grav', 'dist_faults'};   % {} = all evidence layers",
    "cfg.features.as_numeric = {};                     % classified layers used as numbers",
  ];
  if (reg) code.push("cfg.target_field = 'Au_gpt';  cfg.target_label = 'Au (g/t)';   % regression only");
  code.push("cfg.split.method = 'block'; cfg.split.block_size = 20000;   % or 'random' (test_fraction 0.3)");
  if (clf) code.push('cfg.negatives.min_distance = 2000;                 % m from any known deposit');
  code.push('cfg.seed = 42;');
  for (const l of (o.params || [])) code.push(l);
  code.push(clf ? `res = mpm_classification_run(cfg, ${o.key.clf}());` : `res = mpm_regression_run(cfg, ${o.key.reg}());`);
  if (clf && reg) code.push(`% regression: res = mpm_regression_run(cfg, ${o.key.reg}());`);
  B.push(['code', code]);
  B.push(['p', 'A saved run can be reopened with `mpm_setup; mpm_results_viewer` (choose `result.mat`); several runs are compared with `mpm_compare_viewer`. ' +
    (clf ? 'To compare methods on the same split, add the method to `classification/run_compare_classification.m`.' : '')]);

  // ------------------------------------------------------------------ 6.3 test
  B.push(['h2', '6.3 Test the method and check the results']);
  const chk = [];
  if (clf) {
    chk.push('**Summary tab:** training and test AUC on samples and on the map (deposit cells vs. all other cells). A large gap between training and test AUC indicates overfitting; compare with the cross-validated AUC in the log.');
    chk.push('**Success-rate & P–A tab:** the P–A intersection gives the share of deposits captured in the corresponding share of the area; the efficiency (captured % / area %) should be clearly above 1.');
    chk.push('**Map tab:** high scores should follow geologically sensible features, not artefacts (grid edges, data-coverage boundaries).');
  }
  if (reg) chk.push('**Regression:** R², RMSE and bias on test samples (model and grade scale), the observed-vs-predicted plot, residual maps and the cut-off ROC/AUC.');
  chk.push('**Importance tab:** the most important layers should be explainable by the deposit model.');
  chk.push('**Tuning tab:** if the best value lies at the edge of the grid, extend the grid (`cfg.tune.grid`) and run again.');
  for (const c of (o.checks || [])) chk.push(c);
  B.push(['bullets', chk]);
  B.push(['p', '**Robustness tests** (recommended before reporting results):']);
  B.push(['bullets', [
    'Repeat the run with several seeds (`cfg.seed = 1, 2, …`); each seed gives a different training/test split. Report mean ± standard deviation of the test AUC (as in the M5 comparison).',
    'Use a spatial block split (`cfg.split.method = \'block\'`, `cfg.split.block_size` larger than the spatial correlation of the evidence) to test transfer to unexplored ground.',
    'Leave out one layer at a time (`cfg.exclude_layers`) to see how much the result depends on it (as done for geology in M7).',
  ]]);
  if (o.verify && o.verify.length) {
    B.push(['p', '**Verification of the implementation** (Section 5) can be repeated as follows:']);
    B.push(['bullets', o.verify]);
  }
  B.push(['p', '**Reproducing the demo:** run the script unchanged (`dataset = \'carlin\'`, `cfg.seed = 42`); the results should match Section 5 (small differences are possible with another MATLAB version' + (o.gpuNote ? ' or on a GPU' : '') + ').']);
  return B;
}

module.exports = { howto };
