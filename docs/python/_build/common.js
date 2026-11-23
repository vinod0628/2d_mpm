// Shared chapters and result helpers of the Python method documents (read results/*.json from experiments.py).
const fs = require('fs');
const path = require('path');

const RES = path.join(__dirname, 'results');
const load = (name) => JSON.parse(fs.readFileSync(path.join(RES, name + '.json'), 'utf8'));
const has = (name) => fs.existsSync(path.join(RES, name + '.json'));

const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const g = (v, d = 4) => (v === null || v === undefined ? '—' : Number(v).toPrecision(d).replace(/\.?0+e/, 'e'));
const int = (v) => (v === null || v === undefined ? '—' : Math.round(v).toLocaleString('en-US'));
const pm = (m, s, d = 3) => (s === null || s === undefined ? f(m, d) : `${f(m, d)} ± ${f(s, d)}`);
const esc = (s) => String(s).replace(/\$/g, '\\$').replace(/\*/g, '\\*').replace(/`/g, "'");

const DS = { carlin: 'Carlin (Au)', curnamona: 'Curnamona (REE)' };

// ------------------------------------------------------------------ demo data chapter
function demoData(n, protocolRow) {   // protocolRow: optional [label, Carlin, Curnamona] replacing the protocol row
  const c = load('wofe_carlin').dataset_info;
  const r = load('wofe_curnamona').dataset_info;
  const uc = load('univariate_carlin');
  const ur = load('univariate_curnamona');
  const uRows = (u, k) => u.slice(0, k).map((x) => [x.layer, x.group, f(x.auc), `${f(x.ci_low)}–${f(x.ci_high)}`,
    x.auc >= 0.5 ? 'high values' : 'low values', g(x.dep_median, 3), g(x.bg_median, 3)]);
  return [
    ['h2', `${n}.1 Demo projects`],
    ['p', 'Two open datasets of contrasting size and deposit type are shipped with MPM Studio (`demo_data/`) and used for every method, so the method documents can be compared directly.'],
    ['table', {
      caption: `Table ${n}.1. The demo projects.`,
      head: ['', 'Carlin (Nevada, USA)', 'Curnamona (South Australia)'],
      widths: [2626, 3200, 3200],
      rows: [
        ['Deposit type', 'Carlin-type (sediment-hosted, disseminated) Au', 'REE (with U–Th) occurrences'],
        ['Source', 'USGS Open-File Report 01-221 (Raines, 2001)', 'DEEP-SEAM v1.0 open dataset (northern Curnamona)'],
        ['Coordinate system', 'Lambert conformal conic (Clarke 1866), m', r.crs || 'EPSG:28354 (GDA94 / MGA 54)'],
        ['Raster cell', `${f(c.cell_m, 0)} m`, `${f(r.cell_m, 0)} m`],
        ['Grid (columns × rows)', `${c.width} × ${c.height}`, `${r.width} × ${r.height}`],
        ['Study area', `${int(c.study_cells)} cells = ${int(c.study_km2)} km²`, `${int(r.study_cells)} cells = ${int(r.study_km2)} km²`],
        ['Known deposits (points / cells)', `${c.deposit_points} / ${c.deposit_cells}`, `${r.deposit_points} / ${r.deposit_cells}`],
        ['Deposit density', `1 per ${int(c.study_km2 / c.deposit_cells)} km²`, `1 per ${int(r.study_km2 / r.deposit_cells)} km²`],
        protocolRow || ['Validation protocol', '5 + 10 spatially stratified 70 / 30 splits', 'leave-one-deposit-out (7 folds)'],
      ],
    }],
    ['h3', `${n}.1.1 Carlin-type gold — geological setting and mineral system`],
    ['p', 'Carlin-type gold deposits of north-central Nevada are Eocene (≈ 42–34 Ma) sediment-hosted, disseminated deposits in which sub-micron gold is carried in arsenian pyrite rims (Hofstra and Cline, 2000; Cline et al., 2005). Ore fluids were reduced, low-salinity, moderately acidic and CO₂-bearing; they rose along **high-angle faults** and spread into **reactive, permeable silty carbonate rocks** of the Palaeozoic passive-margin (lower-plate) sequence, where decarbonatisation, argillisation and silicification consumed acidity and caused sulfidation of host-rock iron and gold precipitation (Cline et al., 2005; Muntean et al., 2011). The deposits are clustered along regional trends that follow deep crustal structures. The diagnostic geochemical halo is **As–Sb–Hg–Tl** (with low base metals), which disperses into stream sediments.'],
    ['p', 'The mineral system therefore predicts four mappable proxies: (1) proximity to faults (fluid pathways), (2) favourable host stratigraphy (chemical trap), (3) As and Sb anomalies in stream sediments (fluid footprint and dispersion from altered/mineralised rock) and (4) low K and Th in airborne radiometrics, because carbonate hosts are poor in K and Th compared with siliciclastic and igneous rocks.'],
    ['h3', `${n}.1.2 REE in the northern Curnamona Province — geological setting and mineral system`],
    ['p', 'The northern Curnamona Province (Mount Painter and Mount Babbage Inliers) contains Mesoproterozoic A-type, high-heat-production granites that are strongly enriched in U, Th and the REE (Neumann et al., 2000). Palaeozoic hydrothermal and metasomatic events remobilised these elements into breccias and veins with allanite, monazite and xenotime (Elburg et al., 2012). REE occurrences are therefore expected where **U–Th–REE-rich granite and its alteration** crop out or lie under thin cover, which airborne gamma-ray spectrometry maps directly (high Th and U, and K from K-feldspar-rich granite and K-metasomatism). High-field-strength elements such as **Nb** are co-enriched with the REE in such A-type systems, while **quartz-rich cover** (dune sand, silcrete, quartzose sediments, seen as a high ASTER silica index) hides the basement and lowers the chance of a known occurrence.'],
    ['h2', `${n}.2 Evidence screening (single-layer AUC)`],
    ['p', 'Before modelling, every evidence layer of the two projects was screened by its single-layer ROC AUC: the probability that a deposit cell has a higher value than a background cell (DeLong et al., 1988; script `docs/python/_build/univariate.py`). AUC > 0.5 means deposits sit on high values, AUC < 0.5 on low values; the further from 0.5, the stronger the association. The screening confirms the mineral-system proxies above and was used to choose the evidence of the Curnamona models.'],
    ['table', {
      caption: `Table ${n}.2. Strongest single-layer associations — Carlin (${c.deposit_cells} deposit cells).`,
      head: ['Layer', 'Group', 'AUC', '95 % CI', 'Deposits on', 'Median at deposits', 'Median background'],
      widths: [1826, 1300, 800, 1300, 1200, 1300, 1300],
      rows: uRows(uc, 12),
    }],
    ['table', {
      caption: `Table ${n}.3. Strongest single-layer associations — Curnamona (${r.deposit_cells} deposit cells; wide CIs because of the small sample).`,
      head: ['Layer', 'Group', 'AUC', '95 % CI', 'Deposits on', 'Median at deposits', 'Median background'],
      widths: [2226, 1200, 700, 1300, 1100, 1250, 1250],
      rows: uRows(ur, 12),
    }],
    ['p', `In Carlin, Sb in stream sediments is the strongest single proxy (AUC ${f(uc.find((x) => x.layer === 'sb_log_idw').auc)} for the log-IDW surface), followed by the favourable host rocks (${f(uc.find((x) => x.layer === 'geology_host').auc)}) and As (${f(uc.find((x) => x.layer === 'as_log_idw').auc)}); deposits lie close to faults (distance AUC ${f(uc.find((x) => x.layer === 'dist_faults').auc)}, i.e. ${f(1 - uc.find((x) => x.layer === 'dist_faults').auc)} for proximity) and on low radiometric Th and K, as expected for carbonate hosts. In Curnamona, the radiometric U, total dose and Th channels dominate (AUC ${f(ur.find((x) => x.layer === 'rad_u').auc)}, ${f(ur.find((x) => x.layer === 'rad_dose').auc)}, ${f(ur.find((x) => x.layer === 'rad_th').auc)}), followed by K, Nb geochemistry and a low silica index — exactly the U–Th–REE granite signature; distances to faults and granite outlines and the La geochemistry carry almost no signal at the occurrence scale.`],
  ];
}

// ------------------------------------------------------------------ protocol + measures
function protocol(n) {
  return [
    ['h2', `${n}.3 Experimental protocol`],
    ['p', 'All runs use the same code path as the graphical user interface (`REGISTRY[key].run(params)`), driven by `docs/python/_build/experiments.py`, so every number below can be reproduced from the menus with the listed settings.'],
    ['bullets', [
      '**Carlin (34 deposit cells).** Each candidate setting is evaluated on 5 spatially stratified 70 / 30 splits of the deposits (seeds 100–104). The setting with the highest mean **test** ROC AUC is selected; settings within 0.005 of the best are treated as equivalent (the differences are far below the split-to-split s.d.) and the earliest of them in the candidate list — the default or simpler setting — is chosen. It is then re-evaluated on **10 independent splits** (seeds 0–9; mean ± s.d. reported) and run once with seed 42 for the maps, curves and tables.',
      '**Curnamona (7 deposits).** A 30 % hold-out would leave two test deposits, so **leave-one-deposit-out (LODO)** validation is used: the model is trained on six occurrences and the held-out occurrence is ranked against all background cells; its percentile is the AUC of one positive, averaged over the seven folds. The reference run uses all seven occurrences (training AUC only).',
      'Spatially stratified split: deposits ordered along a Hilbert curve, one test deposit drawn per run of consecutive deposits (Stevens and Olsen, 2004). Because 15 of the 34 Carlin deposits form one dense cluster, test deposits often have training neighbours a few cells away, so the test AUCs are an upper bound of what a spatial block hold-out would give (Roberts et al., 2017).',
      'Unit cell 1 km², study-area mask = the project mask, cells with missing evidence excluded (logistic regression, random forest) or given weight 0 (weights of evidence) as noted per method.',
    ]],
    ['h2', `${n}.4 Performance measures`],
    ['bullets', [
      '**ROC AUC** — Mann–Whitney probability that a deposit cell scores higher than a background cell (ties count ½), against **all** other study-area cells; 95 % confidence interval from the DeLong et al. (1988) variance (Fawcett, 2006).',
      '**Success / prediction-rate curve** — share of training / test deposits captured against the share of the study area classified as prospective, highest scores first; its area (SR-AUC) equals the ROC AUC up to the deposit cells (Chung and Fabbri, 2003).',
      '**Prediction–area (P–A) plot** — prediction rate and occupied area against the score threshold; at the intersection, *x* % of the deposits fall in (100 − *x*) % of the area; a higher *x* means a better map (Yousefi and Carranza, 2015).',
      '**TPR at FPR** — share of deposits captured when 5, 10, 20 or 30 % of the background is selected (the operational targeting budget).',
    ]],
  ];
}

// ------------------------------------------------------------------ result tables
function searchTable(J, caption) {
  if (J.protocol === 'expert') {
    const bool = J.search.some((x) => x.fav_ratio !== undefined);
    return ['table', {
      caption,
      head: bool ? ['Configuration', 'ROC AUC of the condition count', 'Favourable area %', 'Deposits in it %', 'Deposit / area ratio']
        : ['Configuration', 'ROC AUC (all deposits)', 'P–A: deposits % at the intersection', 'Time (s)'],
      widths: bool ? [3326, 1500, 1400, 1400, 1400] : [4026, 1800, 1800, 1400],
      rows: J.search.map((x) => [(x.label === J.selected.label ? '**' + x.label + '** (selected)' : x.label), f(x.mean_test_auc)]
        .concat(bool ? [f(x.fav_area_pct, 1), f(x.fav_deposits_pct, 1), f(x.fav_ratio, 2)] : [f(x.pa_prediction, 1), f(x.seconds, 1)])),
    }];
  }
  const lodo = J.protocol === 'lodo';
  return ['table', {
    caption,
    head: ['Candidate setting', lodo ? 'Mean LODO AUC ± s.d.' : 'Mean test AUC ± s.d. (5 splits)', 'Mean training AUC', 'Time per run (s)'],
    widths: [3726, 2300, 1600, 1400],
    rows: J.search.map((s) => [(s.label === J.selected.label ? '**' + s.label + '** (selected)' : s.label),
      pm(s.mean_test_auc, s.sd_test_auc), f(s.mean_train_auc), f(s.seconds, 1)]),
  }];
}

function robustTable(J, caption) {
  const R = J.robust_summary;
  if (!R) return null;
  return ['table', {
    caption,
    head: ['Measure (10 splits, seeds 0–9)', 'Mean ± s.d.', 'Min – max'],
    widths: [4026, 2500, 2500],
    rows: [
      ['Training ROC AUC', pm(R.train_auc_mean, R.train_auc_sd), '—'],
      ['Test ROC AUC', pm(R.test_auc_mean, R.test_auc_sd), `${f(R.test_auc_min)} – ${f(R.test_auc_max)}`],
      ['Test success-rate AUC', f(R.test_sr_mean), '—'],
      ['Test P–A intersection (deposits %)', f(R.test_pa_mean, 1), '—'],
      ['Run time (s)', f(R.seconds_mean, 1), '—'],
    ],
  }];
}

function curvesTable(J, caption) {
  const rows = Object.entries(J.reference.curves).map(([k, c]) => [k, c.deposits, int(c.negatives), f(c.auc),
    `${f(c.auc_lo)}–${f(c.auc_hi)}`, f(c.sr_auc), c.pa_prediction == null ? '—' : `${f(c.pa_prediction, 1)} / ${f(c.pa_area, 1)}`,
    ['0.05', '0.1', '0.2', '0.3'].map((x) => f(c.tpr_at[x], 2)).join(' / ')]);
  return ['table', {
    caption,
    head: ['Set', 'Deposit cells', 'Background cells', 'ROC AUC', '95 % CI (DeLong)', 'SR-AUC', 'P–A: deposits / area %', 'TPR at FPR 5 / 10 / 20 / 30 %'],
    widths: [900, 900, 1200, 900, 1200, 900, 1326, 1700],
    rows,
  }];
}

function lodoTable(J, caption) {
  const s = J.search.find((x) => x.label === J.selected.label);
  if (!s || !s.lodo) return null;
  return ['table', {
    caption,
    head: ['Held-out occurrence', 'Cell (row, col)', 'LODO AUC (percentile of the held-out occurrence)', 'Training AUC (6 occurrences)'],
    widths: [1800, 1800, 3226, 2200],
    rows: s.lodo.map((x) => [`#${x.deposit + 1}`, `${x.row}, ${x.col}`, f(x.lodo_auc), f(x.train_auc)]),
  }];
}

function evidenceTable(J, caption) {
  return ['table', {
    caption,
    head: ['Evidence layer', 'Used as', 'Description', 'Values'],
    widths: [2100, 1500, 4026, 1400],
    rows: J.evidence.map((e) => [e.name.replace('@q8', ' (8 classes)'), e.kind, esc(e.description),
      e.kind === 'continuous' ? `${g(e.min, 3)} – ${g(e.max, 3)}` : `${e.n_values} classes`]),
  }];
}

// knowledge-driven methods: nothing is trained, every deposit validates
function protocolKD(n, missing) {
  const all = protocol(n);
  const i = all.findIndex((b) => b[0] === 'h2' && String(b[1]).startsWith(`${n}.4`));
  return [
    ['h2', `${n}.3 Experimental protocol`],
    ['p', 'All runs use the same code path as the graphical user interface (`REGISTRY[key].run(params)`), driven by `docs/python/_build/experiments.py`.'],
    ['bullets', [
      '**No training.** The memberships, conditions and operators encode the deposit model; deposits are not used to build the map, so **all** deposits (34 Carlin, 7 Curnamona) validate it — there is no train / test split.',
      '**Configurations.** The evidence of each project is assigned to the components of its mineral system (Carlin: trap = favourable host, pathway = distance to faults, footprint = As and Sb; Curnamona: source = radiometric Th and U, enrichment = Nb, exposure = silica index). Alternative operators, γ values and thresholds are compared by the ROC AUC of all deposits; configurations within 0.005 of the best are equivalent and the earliest (default / simplest) is chosen.',
      '**Caveat.** Choosing among configurations with the deposits makes the selected map slightly optimistic; the differences between reasonable configurations are small (tables below), which is itself the main result: a well-founded deposit model matters more than the operator.',
      missing || 'Study-area mask = the project mask; cells with missing evidence excluded.',
    ]],
    ...all.slice(i),
  ];
}

const MISSING_RENORM = 'Study-area mask = the project mask; where a layer has no data (e.g. outside the mapped host units of Carlin) the weights are renormalised over the layers with data.';

module.exports = { MISSING_RENORM, protocolKD, load, has, f, g, int, pm, esc, DS, demoData, protocol, searchTable, robustTable, curvesTable,
  lodoTable, evidenceTable };
