const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('kt_rk_carlin');
const K = C.load('kt_rk_curnamona');
const st = (X) => X.reference.stats;
const ho = (X) => X.holdout;
const sel = (X) => X.search.find((s) => s.label === X.selected.label);
const cand = (X, start) => X.search.find((s) => s.label.startsWith(start));
const dauc = (X) => (X.reference.curves.training ? X.reference.curves.training.auc : null);
const nug = (X) => st(X).nugget / (st(X).nugget + st(X).sill);
let tn = 3;
const T = () => ++tn;

const targetTable = (X, cap) => ['table', {
  caption: cap, head: ['Item', 'Value'], widths: [3826, 5200], rows: [
    ['Observations (layer / field)', `${X.target.layer} / ${X.target.field} — ${X.target.description}`],
    ['Samples / with a valid value', `${int(X.target.samples)} / ${int(X.target.samples_valid)}`
      + (X.target.zero_is_missing ? ` (${int(X.target.zeros)} zeros = below detection or not analysed, set to NoData)` : '')],
    ['Sampled cells (values averaged per cell)', `${int(X.target.cells)}`],
    ['Mean sample spacing √(study area / cells)', `${f(X.target.spacing_km, 1)} km`],
    ['Min / median / 95th percentile / max', `${g(X.target.min, 3)} / ${g(X.target.median, 3)} / ${g(X.target.p95, 3)} / ${g(X.target.max, 3)} ${X.target.unit}`],
    ['Log offset c = 0.1 × median of the positive values', `${g(st(X).c, 3)} ${X.target.unit}`],
  ] }];

const knowTable = (X, cap) => ['table', {
  caption: cap, head: ['Knowledge set', 'Layer', 'Process', 'Direction / class scores'], widths: [2600, 2400, 1500, 2526],
  rows: Object.entries(X.knowledge).flatMap(([k, rows]) => rows.map((r, i) => [i === 0 ? k : '', r.layer, r.process, r.direction])),
}];

const searchTable = (X, cap) => ['table', {
  caption: cap,
  head: ['Candidate setting', 'CV RMSE ± s.d. (selected α, s)', 'α / s', 'CV RMSE mean + kriging only', 'ℓ (× ranges)', 'Nugget share', 'Deposit AUC', 'Time (s)'],
  widths: [2826, 1300, 700, 1100, 800, 800, 800, 700],
  rows: X.search.map((s) => [(s.label === X.selected.label ? `**${s.label}** (selected)` : s.label), pm(s.cv_rmse, s.cv_rmse_sd),
    `${g(s.alpha, 2)} / ${g(s.shrink, 2)}`, f(s.cv_rmse_kriging_only), f(s.length_scale, 2), f(s.nugget / (s.nugget + s.sill), 2),
    f(s.deposit_auc), f(s.seconds, 0)]),
}];

const cvTable = (X, cap) => {
  const cv = X.reference.cv;
  const al = [...new Set(cv.map((r) => r.alpha))].sort((a, b) => a - b);
  const sh = [...new Set(cv.map((r) => r.shrink))].sort((a, b) => a - b);
  return ['table', {
    caption: cap, head: ['α (knowledge trend) \\ s (forest)'].concat(sh.map((s) => `s = ${g(s, 2)}`)), widths: [2826, 2066, 2067, 2067],
    rows: al.map((a) => [`α = ${g(a, 2)}`].concat(sh.map((s) => {
      const r = cv.find((x) => x.alpha === a && x.shrink === s);
      const t = pm(r.cv_rmse, r.cv_rmse_sd);
      return r.selected ? `**${t}** (selected)` : (r.within_one_se ? `${t} (within 1 SE)` : t);
    }))),
  }];
};

const holdTable = (X, cap) => ['table', {
  caption: cap, head: ['Measure (20 % of the sampled cells held out, model on 80 %)', 'Value'], widths: [5826, 3200], rows: [
    ['Training / test cells', `${int(ho(X).train_cells)} / ${int(ho(X).test_cells)}`],
    ['RMSE of ln(y + c): KT-RK estimate', f(ho(X).rmse_log)],
    ['RMSE: trend only (knowledge + forest, without kriging)', f(ho(X).rmse_trend)],
    ['RMSE: mean of the training values (no model)', f(ho(X).rmse_mean)],
    ['R² of the estimate (log scale)', f(ho(X).r2)],
    ['Pearson r (log scale) / Spearman ρ (values)', `${f(ho(X).pearson)} / ${f(ho(X).spearman_value)}`],
  ] }];

const interp = (X, name) => {
  const s = st(X);
  const parts = [];
  parts.push(`The selected setting is **${X.selected.label}**: cross-validated RMSE of ln(y + c) **${f(s.cv_rmse)}** against ${f(s.cv_rmse_mean_only)} for the mean plus kriging alone (α = s = 0).`);
  if (s.alpha > 0) parts.push(`The knowledge trend is used (α = ${g(s.alpha, 2)}, slope α·b1 = ${f(s.b1)}): the expert favourability explains part of the ${X.target.element} distribution (r = ${f(s.corr_k_z)} with the log values).`);
  else parts.push(`The one-standard-error rule set the knowledge trend to α = 0 (r = ${f(s.corr_k_z)} between K and the log values)${s.shrink > 0 ? `, while the forest correction is kept (s = ${g(s.shrink, 2)}): the forest sees the same proxies, scaled in the expert direction, and represents their non-linear, interacting effect on ${X.target.element} better than one linear slope on K` : ': neither the knowledge trend nor the forest improves on kriging alone'}.`);
  parts.push(`The fitted covariance is ${nug(X) > 0.85 ? 'dominated by the nugget' : nug(X) > 0.55 ? 'mostly nugget' : nug(X) > 0.35 ? 'about half nugget' : 'mostly spatially structured'} (nugget share ${f(nug(X), 2)}), so ${nug(X) > 0.85 ? `the kriged residual is small (Figure ${name}d) and the estimate is close to the trend; the large nugget reflects short-scale variability (sample support, local lithology) that no 2D model can resolve at a ${f(X.target.spacing_km, 1)} km sample spacing` : `the kriging adds a clear local correction near the samples (Figure ${name}d) and returns to the trend between them`}.`);
  parts.push(`On the 20 % hold-out the estimate reaches RMSE ${f(ho(X).rmse_log)} (trend alone ${f(ho(X).rmse_trend)}, mean ${f(ho(X).rmse_mean)}), R² ${f(ho(X).r2)} and a rank correlation of ${f(ho(X).spearman_value)} with the measured values.`);
  return parts.join(' ');
};

const strikeWhy = () => {
  const jg = cand(J, 'host + faults + As, strike 13°'), jt = cand(J, 'host + faults + As, strike 325°'), ji = cand(J, 'host + faults + As, isotropic');
  const kg = cand(K, 'Th + U + granite + silica, strike 70°'), kc = cand(K, 'Th + U + granite + silica, strike 160°');
  const spread = Math.max(jg.cv_rmse, jt.cv_rmse, ji.cv_rmse) - Math.min(jg.cv_rmse, jt.cv_rmse, ji.cv_rmse);
  return `Candidates from geology: Carlin — the NNE fault grain (13°, Basin-and-Range faults) and the NW Carlin trend (325°, alignment of the deposits over deep crustal structures); Curnamona — the ENE fault grain (70°). `
    + `For Carlin stream-sediment Sb the direction hardly matters (CV RMSE ${f(jg.cv_rmse)}, ${f(jt.cv_rmse)} and isotropic ${f(ji.cv_rmse)}; spread ${f(spread)}): Sb is dispersed along the drainage network, which crosses the structures, so the residual has little directional continuity; ${f(J.selected.strike, 0)}° is the earliest candidate within 0.005 of the best. `
    + `For Curnamona La the fault grain (${f(kg.cv_rmse)}) beats the cross direction (${f(kc.cv_rmse)}): the surface samples sit on the basement, whose granites, alteration and REE-bearing breccias are elongated along the ENE structures.`;
};

const aniso = (X, grain, cross, iso) => {
  const a = cand(X, grain), c = cross ? cand(X, cross) : null, i = iso ? cand(X, iso) : null;
  const t = [];
  const TIE = 0.005;                        // the selection rule: differences below 0.005 are ties
  const cmp = (x, y) => (x < y - TIE ? 'better than' : x > y + TIE ? 'worse than' : 'equivalent to (within 0.005)');
  if (a && c) t.push(`Continuity along the fault grain (CV RMSE ${f(a.cv_rmse)}) is ${cmp(a.cv_rmse, c.cv_rmse)} the alternative direction (${f(c.cv_rmse)})`);
  if (a && i) t.push(`${t.length ? 'and ' : 'Continuity along the fault grain is '}${cmp(a.cv_rmse, i.cv_rmse)} an isotropic covariance (${f(i.cv_rmse)})`);
  return t.length ? t.join(', ') + '.' : '';
};

const kRows = (X) => X.knowledge[X.selected.know].map((r) => `${r.layer} → ${r.process}, ${r.direction === '+' ? 'high values favourable' : r.direction === '−' ? 'low values favourable' : 'scores ' + r.direction}, weight 1`);
const setupTable = (cap) => ['table', {
  caption: cap, head: ['Dialog field', 'Carlin (Sb)', 'Curnamona (La)'], widths: [2626, 3200, 3200], rows: [
    ['Project (*Project ▸ Open demo project*)', 'Carlin-type Au, Nevada (USGS)', 'REE, Curnamona, South Australia'],
    ['Data preparation', `In ${J.target.layer}, set ${J.target.field} = 0 to NoData (${int(J.target.zeros)} samples: below detection / not analysed) — in a GIS before loading (the attribute table of MPM Studio is read-only); the experiments do this in experiments_ktrk.py`, 'none'],
    ['Deposits (validation only)', 'deposits', 'deposits'],
    ['Study area mask', 'study_mask', 'study_mask'],
    ['Absence points (ROC)', '— all background cells —', '— all background cells —'],
    ['Knowledge table (ticked rows)', kRows(J).join('; '), kRows(K).join('; ')],
    ['Process weights', 'all 1', 'all 1'],
    ['Observations (points) / value field', `${J.target.layer} / ${J.target.field}`, `${K.target.layer} / ${K.target.field}`],
    ['Strike (° from north)', f(J.selected.strike, 0), f(K.selected.strike, 0)],
    ['Range along / across strike (m)', `${int(J.selected.along)} / ${int(J.selected.across)}`, `${int(K.selected.along)} / ${int(K.selected.across)}`],
    ['Log offset c', `auto (→ ${g(st(J).c, 3)})`, `auto (→ ${g(st(K).c, 3)})`],
    ['Cross-validation folds', '10', '10'],
    ['Max. points for the covariance fit', '1500', '1500'],
    ['Output process maps / prefix', 'off / ktrk', 'off / ktrk'],
    ['Resample evidence to the first grid', 'on', 'on'],
    ['Selected by the tool', `α = ${g(st(J).alpha, 2)}, s = ${g(st(J).shrink, 2)}`, `α = ${g(st(K).alpha, 2)}, s = ${g(st(K).shrink, 2)}`],
    ['Run time', `${f(J.reference.seconds, 0)} s`, `${f(K.reference.seconds, 0)} s`],
  ] }];

module.exports = {
  file: 'kt_rk.docx',
  code: 'KT-RK',
  short: 'Knowledge-trend regression-kriging',
  title: 'Knowledge-Trend Regression-Kriging (KT-RK) of a Continuous Exploration Target',
  subtitle: 'Hybrid method M5 of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'KT-RK is method M5 of the improved hybrid MPM methods proposed for MPM Studio (MPM Studio project, 2026). It estimates a **continuous** quantity measured at points — composited drill-hole grade, thickness or accumulation (grade × thickness), or a geochemical concentration — by regression-kriging (Odeh et al., 1995; Hengl et al., 2007) whose trend is built from mineral-system knowledge: a sign-constrained trend on the expert favourability *K*, a random-forest correction of that trend (Breiman, 2001; Hengl et al., 2018) and anisotropic kriging of the remaining residual along the expert strike. Both data-driven parts are shrunk toward the simpler model by the one-standard-error rule, so the expert trend and the forest are only used where cross-validation supports them. MPM Studio: *Hybrid ▸ Knowledge prior + data correction ▸ Knowledge-Trend Regression-Kriging (KT-RK)*.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4200, 4826], rows: [
      ['`python/logic/hybrid/kt_rk/model.py`', 'Log transform, knowledge trend, forest, one-SE selection, anisotropic covariance, memory-lean kriging'],
      ['`python/logic/hybrid/kt_rk/plugin.py`, `dialogs.py`', 'Samples → cells (per-cell mean), outputs, dialog (value field, strike, ranges)'],
      ['`python/logic/hybrid/common/knowledge.py`, `pipeline.py`, `dialogs.py`', 'Knowledge network and table shared by all hybrid methods'],
      ['`python/tests/test_kt_rk.py`', 'Rotation, kriging = scikit-learn GP, trend selection, one-SE rule, sign constraint, compositing'],
      ['`docs/python/_build/experiments_ktrk.py`', 'Demo experiments of this document'],
    ] }],
    ['h2', '1.1 KT-RK compared with the classification tools'],
    ['p', 'The other MPM Studio tools (weights of evidence, logistic regression, SVM, random forest, MS-KPRB, KP-RF …) answer *where is a deposit likely?* from deposit / non-deposit labels. KT-RK answers *how large is a measured quantity here?* from measured values. Its map is in the units of the target and is validated by prediction errors, not by ROC curves.'],
    ['table', { caption: 'Table 1.2. KT-RK and the classification tools.', head: ['', 'Classification tools (SVM, RF, …)', 'KT-RK'], widths: [1800, 3600, 3626], rows: [
      ['Question', 'Where is a deposit likely?', 'How large is grade / thickness / concentration here?'],
      ['Training data', 'Deposit points (1) and sampled non-deposit cells (0)', 'Points with a numeric value (≥ 25–30); deposits not needed'],
      ['Main output', 'Prospectivity score or probability 0–1 (ranks cells)', '`prediction` in the target units (e.g. g/t, m, ppm)'],
      ['Uncertainty', 's.d. over the ensemble', '`log_sd` = kriging s.d.: small near samples, large far from them'],
      ['Validation', 'ROC AUC, success rate, P–A against deposits', 'Cross-validated RMSE of ln(y + c); deposits optional (ROC of the estimated map)'],
      ['Far from data', 'The model is applied everywhere', 'Returns to the knowledge + forest trend'],
      ['Exploration stage', 'Regional targeting', 'Prospect scale: where grade or thickness is likely higher, where to drill next'],
    ] }],
    ['note', 'KT-RK is a 2D estimate for targeting, **not a resource estimate**: one composited value per hole, point support (no change of support to blocks), a median-type back-transform, no top-cutting, declustering or estimation domains, and random (not spatial) cross-validation. It is not suitable for JORC or NI 43-101 reporting (Journel and Huijbregts, 1978).'],
    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Regression-kriging'],
    ['p', 'A regionalised variable is split into a deterministic trend explained by covariates and a spatially correlated residual (Matheron, 1963; Cressie, 1993). Regression-kriging estimates the trend by regression on the covariates and adds the simple kriging of the regression residuals (Odeh et al., 1995; Hengl et al., 2007). It is equivalent to universal kriging with external drift when the trend is linear, but allows any regression model — here a knowledge trend and a random forest.'],
    ['h2', '2.2 The knowledge trend'],
    ['p', 'The covariate of the trend is the expert favourability *K* of the mineral-system knowledge network shared by all hybrid methods: every evidence layer is a proxy of one process (source, pathway, trap, deposition, …), robustly scaled in its expert direction and turned into a membership; the proxies of a process are combined OR-like (power mean), the processes AND-like (weighted geometric mean). *K* is built **without** the target values. Its slope is constrained to be ≥ 0: if the data run against the expert model, the trend is switched off instead of reversing the geological meaning (MPM Studio project, 2026).'],
    ['h2', '2.3 The forest correction'],
    ['p', 'A random forest on the robust-scaled proxies (categorical layers: their expert class score) models the part of the target the linear knowledge trend cannot represent — non-linear responses, interactions, thresholds. It is fitted to the residuals of the knowledge trend and its out-of-bag predictions are used, so that the residuals passed to kriging are honest. Its weight *s* ∈ {0, ½, 1} shrinks it toward the trend.'],
    ['h2', '2.4 Why the residual is kriged'],
    ['p', 'The trend only explains what the maps explain. Grade and thickness also depend on things no evidence map shows — ore shoots, local thickening, blind structures, sampling and assay effects — and these residuals are spatially correlated: a hole richer than its trend usually has rich neighbours, especially along the strike of the controlling structure or bed. Kriging the residual makes the estimate honour the measured values near the observations and fall back to the knowledge + forest trend far from them, so the data correction is limited by the distance to the data — the design rule of all improved hybrid methods. If the residuals have no spatial structure, the fitted covariance is almost pure nugget, the kriged residual is close to zero and the result is the trend: kriging costs nothing when it is not needed. The kriging variance gives an uncertainty map that shows where new samples or drill holes would be most informative.'],
    ['h2', '2.5 Geological anisotropy'],
    ['p', 'Ore shoots, beds and dispersion trains are elongated along the controlling structure. The coordinates are therefore rotated to the expert strike and divided by the expert ranges along and across strike before an isotropic covariance is fitted. Because the scale ℓ of the covariance is fitted, the ranges set the **ratio** of continuity along and across strike, while ℓ adjusts the overall size within 0.3–3 times the ranges.'],
    ['h2', '2.6 Choosing the simplest adequate model'],
    ['p', 'With tens to a few hundred observations, cross-validated errors of similar models differ by less than their standard error. The one-standard-error rule (Breiman et al., 1984; Hastie et al., 2009) selects the simplest model (smallest α + s) whose cross-validated RMSE is within one standard error of the best, which protects the estimate from over-fitting the knowledge trend or the forest.'],
    ['h1', '3 Mathematical basis'],
    ['p', 'Knowledge favourability (expert parameters, no labels): robust directed scale $u_{ij}=d_{j}(x_{ij}-\\rm{med}_{j})/\\rm{IQR}_{j}$, membership $μ_{ij}=1/(1+e^{-s(u_{ij}-c)})$ with $s=1.5$, $c=0.5$ (categorical layers: the expert class score), process score and favourability'],
    ['eq', 'F_{ik}=(\\sum_{j∈k}{w_{j}μ_{ij}^{p}})^{1/p},      K_{i}=\\prod_{k}{F_{ik}^{v_{k}}},      p=3', 'kt-k'],
    ['p', 'with expert weights $w_{j}$ (normalised within each process) and process weights $v_{k}$ (normalised to 1). Target and model:'],
    ['eq', 'z=\\ln(y+c),      c=0.1·\\rm{median}\\{y>0\\}', 'kt-z'],
    ['eq', 'z(\\bf{x})=b_{0}+α b_{1}K(\\bf{x})+s f_{RF}(\\bf{x})+ε(\\bf{x}),      α,s∈\\{0,0.5,1\\}', 'kt-model'],
    ['eq', 'b_{1}=\\max(\\frac{\\rm{cov}(K,z)}{\\rm{var}(K)},0),      b_{0}=\\bar{z}-α b_{1}\\bar{K}', 'kt-b'],
    ['p', 'The forest $f_{RF}$ (200 trees, minimum leaf 5) is fitted to $z-b_{0}-α b_{1}K$; the kriged residual uses its out-of-bag predictions $f_{RF}^{oob}$:'],
    ['eq', 'r_{i}=z_{i}-b_{0}-α b_{1}K_{i}-s f_{RF}^{oob}(\\bf{x}_{i})', 'kt-r'],
    ['p', 'Anisotropy: with the strike azimuth θ (clockwise from north), unit vectors $\\bf{a}=(\\sin θ,\\cos θ)$ along and $\\bf{b}=(\\cos θ,-\\sin θ)$ across strike and ranges $R_{1}$, $R_{2}$,'],
    ['eq', '\\bf{h}\'=(\\frac{\\bf{h}·\\bf{a}}{R_{1}},\\frac{\\bf{h}·\\bf{b}}{R_{2}}),      C(\\bf{h})=σ^{2}\\exp(-\\frac{‖\\bf{h}\'‖^{2}}{2ℓ^{2}})+τ^{2}δ(\\bf{h})', 'kt-cov'],
    ['p', '$σ^{2}$, $ℓ∈[0.3,3]$ and the nugget $τ^{2}$ maximise the Gaussian marginal likelihood of the residuals (Rasmussen and Williams, 2006). Simple kriging of the zero-mean residual at $\\bf{x}_{0}$, with $\\bf{C}$ the covariance matrix of the observations and $\\bf{c}_{0}$ their covariances with $\\bf{x}_{0}$:'],
    ['eq', '\\hat{r}(\\bf{x}_{0})=\\bf{c}_{0}^{T}\\bf{C}^{-1}\\bf{r},      σ_{K}^{2}(\\bf{x}_{0})=σ^{2}+τ^{2}-\\bf{c}_{0}^{T}\\bf{C}^{-1}\\bf{c}_{0}', 'kt-krige'],
    ['eq', '\\hat{z}=b_{0}+α b_{1}K+s f_{RF}+\\hat{r},      \\hat{y}=e^{\\hat{z}}-c', 'kt-pred'],
    ['p', '$\\hat{y}$ estimates the median, not the mean, of a log-normal target (a mean estimate would add $σ_{K}^{2}/2$ in the exponent); it is meant for ranking and targeting. One-SE rule: with $\\bar{e}(α,s)$ the mean RMSE over the folds and $\\rm{sd}$ its standard deviation, the selected pair minimises $α+s$ subject to $\\bar{e}(α,s)≤\\bar{e}_{min}+\\rm{sd}(e_{min})/\\sqrt{n_{folds}}$.'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Study area and knowledge** — study cells, robust scaling of the evidence on all cells, expert favourability *K* [[kt-k]] (no target values used).',
      '**Observations** — sample points → grid cells; several samples in one cell are averaged (compositing to cell support); values that are missing or negative are skipped; $z=\\ln(y+c)$ [[kt-z]].',
      '**Covariance** — knowledge trend + forest with α = s = 1 on all observations; the anisotropic covariance [[kt-cov]] is fitted to their residuals (on a random subset of at most 1 500 points; the cost of the fit grows with *n*³).',
      '**Cross-validation** — for each of the 9 pairs (α, s) and each of the 10 random folds: trend [[kt-b]] and forest on the training observations, residuals [[kt-r]], kriging with the fitted covariance, RMSE of $z$ in the held-out fold.',
      '**Selection** — one-standard-error rule.',
      '**Final model** — trend and forest on all observations; kriging of the residuals at every study cell [[kt-krige]] in blocks of 2 000 cells (one covariance block of 2 000 × *n* values at a time; identical to scikit-learn\'s Gaussian-process prediction); estimate [[kt-pred]].',
      '**Validation (optional)** — ROC of the estimated map against known deposits.',
    ]],
    ['h1', '5 How to use'],
    ['h2', '5.1 Observations'],
    ['bullets', [
      '**Point layer and numeric field** — composited drill holes (one value per hole: grade, thickness, accumulation; holes that missed the bed get 0), or geochemical samples. At least about 25–30 observations are needed.',
      '**Below-detection and not-analysed values** — set them to NoData before loading the layer (e.g. with the field calculator of a GIS; the attribute table of MPM Studio is read-only) if they are stored as 0 (as in the NURE data); otherwise they are treated as real zeros (the offset *c* softens but does not remove their effect).',
      '**Geological domains** — kriging interpolates the residual across unit contacts. If the target behaves differently in different units, run KT-RK per unit: reclassify the unit into a mask (*Data ▸ Reclassify*), choose it as **Study area mask**, and use only that unit\'s observations.',
    ]],
    ['h2', '5.2 Knowledge table'],
    ['bullets', [
      '**Layers** — 2–5 proxies of the processes that control the **target** (for a grade: the trap and deposition proxies; for a geochemical element: its source and dispersion). Never tick a raster interpolated from the same target values (e.g. the IDW surface of the same element): the trend would reproduce the target.',
      '**Continuous layers** — process and direction (high or low values favourable).',
      '**Categorical layers (e.g. a geology map)** — process (usually *trap*, sometimes *source*) and an expert score 0–1 per unit, e.g. `12=0.9; 7=0.8; 3=0.4; 1=0.1`; units without a score get 0.05; the pre-filled `class=0.5` everywhere means "no information" and must be edited. Scores refer to the target being estimated. The forest sees the **score** of a unit, not its code: units with the same score are indistinguishable for it — give different scores, or add distance / indicator rasters of units as continuous proxies, if the data should learn the effect of each unit.',
      '**Weights** — expert weight of a layer inside its process and process weights (right-hand table).',
    ]],
    ['h2', '5.3 Settings'],
    ['table', { caption: 'Table 5.1. KT-RK settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2300, 1700, 5026], rows: [
      ['Deposits', 'deposit layer if present', 'Validation only (ROC of the estimated map); no train / test split.'],
      ['Study area mask', 'project mask', 'Cells to estimate; also the way to estimate per geological domain.'],
      ['Observations / value field', '—', 'Point layer and numeric attribute.'],
      ['Strike (° from north)', '0', 'Direction of the longest continuity: the structural grain of the controlling faults or the bed. Take it from the fault map (length-weighted axial mean) or from the elongation of known ore shoots.'],
      ['Range along / across strike', '20 / 6 cells', 'Expected continuity of the residual; a few sample spacings along strike, about a third of that across. Only the ratio matters strongly: ℓ is fitted within 0.3–3 × the ranges.'],
      ['Log offset c', 'auto = 0.1 × median', 'Keeps ln defined for zeros; larger *c* compresses the low values.'],
      ['Cross-validation folds', '10', 'Random folds over the observations for the choice of α and s.'],
      ['Max. points for the covariance fit', '1 500', 'Random subset for the marginal-likelihood fit (cost ∝ *n*³); kriging always uses every observation.'],
      ['Output process maps', 'off', 'Process scores of the knowledge network.'],
    ] }],
    ['h2', '5.4 Outputs'],
    ['table', { caption: 'Table 5.2. Output rasters and tables.', head: ['Output', 'Content'], widths: [2300, 6726], rows: [
      ['`prediction`', 'Estimate $\\hat{y}=e^{\\hat{z}}-c$ in the units of the target (median-type).'],
      ['`log_prediction`', '$\\hat{z}$ = trend + kriged residual (log scale).'],
      ['`log_sd`', 'Kriging s.d. (log scale): low near observations, high far from them — where new samples or holes are most informative.'],
      ['`trend`', 'Knowledge + forest trend; `log_prediction − trend` is the local anomaly that only the observations show.'],
      ['`expert`, `limiting_process`, `process_*`', 'Expert favourability, weakest process, process scores.'],
      ['Tables', '`summary`, `cv` (RMSE for every α, s; selected; within one SE), `observations` (value, trend, estimate at every sampled cell).'],
    ] }],
    ['h1', '6 Demo data: settings, results and their geological meaning'],
    ...C.demoData(6, ['Validation (KT-RK)', 'Sb of 3 472 NURE stream-sediment samples: 10-fold CV + 20 % hold-out', 'La of 615 surface samples: 10-fold CV + 20 % hold-out']),
    ['h2', '6.3 Experimental protocol'],
    ['p', 'The demo projects contain no drill-hole grades, so KT-RK is demonstrated on the geochemical samples shipped with them: Sb of the NURE stream sediments for Carlin-type gold (Sb is a core element of the As–Sb–Hg–Tl Carlin halo) and La of the surface samples for the REE system of the northern Curnamona Province. The result is a knowledge-guided concentration map — the surface-geochemistry counterpart of a grade map. All runs use the GUI code path (`REGISTRY["kt_rk"].run`), driven by `docs/python/_build/experiments_ktrk.py`.'],
    ['bullets', [
      '**Candidates** — knowledge sets, strike and ranges (Tables 6.6 and 6.11). The strikes come from the fault maps: length-weighted axial mean 13° (NNE, Basin-and-Range faults) in Carlin and 70° (ENE) in Curnamona. Each candidate is scored by the 10-fold cross-validated RMSE of ln(y + c) of its own selected (α, s); differences below 0.005 are ties and the earliest candidate wins.',
      '**Hold-out** — the selected setting is re-run on a random 80 % of the sampled cells; the other 20 % are estimated and compared with the measured values.',
      '**Deposits** — the known deposits only check whether high estimates coincide with mineralisation (ROC AUC of the estimated map).',
      '**Caveat** — the folds are random, not spatial, and the covariance is fitted once on all observations before the cross-validation; the cross-validated RMSE is therefore somewhat optimistic. The hold-out refits the covariance on the training cells only.',
    ]],
    ['h2', '6.4 Performance measures'],
    ['bullets', [
      '**RMSE of ln(y + c)** — error on the log scale, comparable between candidates of one target; e.g. 0.69 ≈ a factor of 2 between estimate and measurement.',
      '**R², Pearson r** — share of the variance of the held-out log values explained, and their linear correlation; **Spearman ρ** — rank correlation of estimated and measured values (what matters for targeting).',
      '**Baselines** — the mean of the training values (no model), the trend without kriging, and the mean plus kriging only (α = s = 0 in the cross-validation table).',
    ]],
    ['h2', '6.5 Carlin (Au): Sb in stream sediments'],
    targetTable(J, `Table 6.${T()}. Carlin — target.`),
    knowTable(J, `Table 6.${T()}. Carlin — knowledge sets.`),
    searchTable(J, `Table 6.${T()}. Carlin — candidate settings (10-fold CV of ln(Sb + c); nugget share = τ²/(σ² + τ²)).`),
    cvTable(J, `Table 6.${T()}. Carlin — cross-validated RMSE of every (α, s) of the selected setting.`),
    holdTable(J, `Table 6.${T()}. Carlin — 20 % hold-out.`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — KT-RK estimate of Sb (log scale) with the NURE samples and the known deposits.'],
    ['img', J.figures.panels, 600, 'Figure 6.2. Carlin — (a) CV RMSE for (α, s) (* = selected), (b) hold-out: estimated versus measured, (c) kriging s.d., (d) local residual = estimate − trend.'],
    ['p', interp(J, '6.2')],
    ['p', `${aniso(J, 'host + faults + As, strike 13°', 'host + faults + As, strike 325°', 'host + faults + As, isotropic')} Without As in the knowledge table the cross-validated RMSE is ${f(cand(J, 'host + faults,') ? cand(J, 'host + faults,').cv_rmse : null)}: As and Sb are co-dispersed members of the Carlin halo, so the As proxy carries information on Sb that host rock and faults alone do not. The estimated Sb map ranks the deposits with a ROC AUC of **${f(dauc(J))}**: high Sb in the drainage marks the Carlin footprint, although stream-sediment Sb is displaced downstream from its source.`],
    ['h2', '6.6 Curnamona (REE): La in surface samples'],
    targetTable(K, `Table 6.${T()}. Curnamona — target.`),
    knowTable(K, `Table 6.${T()}. Curnamona — knowledge sets.`),
    searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (10-fold CV of ln(La + c)).`),
    cvTable(K, `Table 6.${T()}. Curnamona — cross-validated RMSE of every (α, s) of the selected setting.`),
    holdTable(K, `Table 6.${T()}. Curnamona — 20 % hold-out.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — KT-RK estimate of La (log scale) with the surface samples and the seven REE occurrences.'],
    ['img', K.figures.panels, 600, 'Figure 6.4. Curnamona — (a) CV RMSE for (α, s), (b) hold-out, (c) kriging s.d., (d) local residual.'],
    ['p', interp(K, '6.4')],
    ['p', `${aniso(K, 'Th + U + granite + silica, strike 70°', 'Th + U + granite + silica, strike 160°', 'Th + U + granite + silica, isotropic')} The candidates 4 × 1.5 km and 8 × 3 km give the same result (${f(cand(K, 'Th + U + granite + silica, strike 70° (fault grain)').cv_rmse)} and ${f(cand(K, 'Th + U + granite + silica, strike 70°, 8').cv_rmse)}) because they have the same anisotropy ratio and the fitted ℓ halves for the longer ranges (ℓ = ${f(cand(K, 'Th + U + granite + silica, strike 70° (fault grain)').length_scale, 2)} and ${f(cand(K, 'Th + U + granite + silica, strike 70°, 8').length_scale, 2)}). Adding Nb geochemistry to the knowledge table lowers the RMSE to ${f(cand(K, '… + Nb').cv_rmse)}: Nb and the light REE are co-enriched in the A-type, high-heat-production granites and their alteration. The La map does not rank the seven REE occurrences (ROC AUC ${f(dauc(K))}): the occurrences are U–Th–REE breccias and veins whose surface La signal is local, while the regional La variation follows granite composition and cover.`],
    ['h2', '6.7 Recommended setup for the demo projects'],
    ['p', 'The settings below reproduce the selected results of Sections 6.5 and 6.6 from the KT-RK dialog; Section 6.8 gives the reasoning behind each value.'],
    setupTable(`Table 6.${T()}. KT-RK dialog settings for the demo projects.`),
    ['h2', '6.8 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal KT-RK settings for the demo projects.`,
      head: ['Parameter', 'Carlin (Sb)', 'Curnamona (La)', 'Statistical and geological reasoning'],
      widths: [1500, 1700, 1700, 4126],
      rows: [
        ['Target / transform', `Sb, ln(Sb + ${g(st(J).c, 2)})`, `La, ln(La + ${g(st(K).c, 2)})`, 'Geochemical concentrations are right-skewed and multiplicative (Reimann et al., 2008); the log scale makes residuals near-normal and errors relative. NURE zeros (below detection / not analysed) are missing values, not concentrations.'],
        ['Knowledge layers', J.selected.know, K.selected.know, 'Proxies of the processes that control the target element: for Sb the reactive host, fluid pathways and the co-dispersed As of the Carlin halo; for La the U–Th–REE granite source (radiometric Th and U, distance to granite), cover (silica index) and the co-enriched Nb. No interpolated surface of the target element itself.'],
        ['Strike', `${f(J.selected.strike, 0)}°`, `${f(K.selected.strike, 0)}°`, strikeWhy()],
        ['Ranges along × across', `${f(J.selected.along / 1000, 1)} × ${f(J.selected.across / 1000, 1)} km`, `${f(K.selected.along / 1000, 1)} × ${f(K.selected.across / 1000, 1)} km`, `A few sample spacings (${f(J.target.spacing_km, 1)} and ${f(K.target.spacing_km, 1)} km) along strike, about a third across; the fitted ℓ (${f(st(J).length_scale, 2)}, ${f(st(K).length_scale, 2)}) rescales them.`],
        ['α (knowledge trend)', g(st(J).alpha, 2), g(st(K).alpha, 2), 'Chosen by the one-SE rule; the sign constraint keeps the expert direction. α = 0 does not mean the knowledge is useless — its proxies still feed the forest.'],
        ['s (forest)', g(st(J).shrink, 2), g(st(K).shrink, 2), 'Half weight is typical with a few hundred observations: the forest captures non-linear proxy effects but is shrunk toward the trend to avoid over-fitting.'],
        ['Covariance', `nugget share ${f(nug(J), 2)}`, `nugget share ${f(nug(K), 2)}`, 'A high nugget is short-scale variability (sample support, local lithology, analytical error) below the sample spacing; it limits how much kriging can add.'],
        ['Folds / covariance points', '10 / 1 500', '10 / 1 500', 'Enough folds for a stable RMSE and its standard error; the subset keeps the likelihood fit fast without changing the kriging.'],
      ],
    }],
    ['h2', '6.9 Discussion'],
    ['bullets', [
      `**Accuracy.** Hold-out RMSE of the log values: Carlin ${f(ho(J).rmse_log)} (mean ${f(ho(J).rmse_mean)}), Curnamona ${f(ho(K).rmse_log)} (mean ${f(ho(K).rmse_mean)}); rank correlation with the measured values ${f(ho(J).spearman_value)} and ${f(ho(K).spearman_value)}.`,
      '**Knowledge versus data.** The expert trend, the forest and the kriging are each kept only when cross-validation supports them; the knowledge network also enters through the forest features, so even α = 0 uses the expert directions and scores.',
      '**Kriging.** Its contribution grows as the nugget share falls and the sample spacing shrinks; with dense drill-hole composites along a structure it dominates near the holes, with regional surface samples it is a modest local correction.',
      '**Anisotropy.** Whether a direction matters depends on how the target is dispersed: clearly for basement surface samples (Curnamona, fault grain better than the cross direction), hardly for stream sediments, whose dispersion follows the drainage (Carlin). The fault grain is a sound default; the ranges matter mainly through their ratio, because ℓ is fitted.',
      '**Limits.** 2D, point support, median-type back-transform, random folds — a targeting tool, not a resource estimate.',
      '**Cost.** The kriging variance of every cell costs O(*n*² × cells): about 9 min for 3 437 observations and 909 545 cells (Carlin), 1–2 min for Curnamona; memory stays below 1 GB through the block-wise prediction.',
    ]],
    ['h1', '7 References'],
    ['refs', R('breiman1984', 'breiman2001', 'cline2005', 'cressie1993', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006',
      'hastie2009', 'hengl2007', 'hengl2018', 'hofstra2000', 'hybriddoc', 'journel1978', 'matheron1963', 'muntean2011',
      'neumann2000', 'odeh1995', 'raines2001', 'rasmussen2006', 'reimann2008', 'smith2001')],
  ],
};
