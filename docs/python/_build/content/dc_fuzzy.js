const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('dc_fuzzy_carlin');
const K = C.load('dc_fuzzy_curnamona');
const tab = (X) => X.reference.tables;
const st = (X) => tab(X).stats;
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const cand = (X, start) => { const r = X.search.find((s) => s.label.startsWith(start)); return r ? r.mean_test_auc : null; };
const sel = (X) => X.selected;
const cal = (X) => ((sel(X).calibration || 'wofe') === 'wofe' ? 'weights of evidence per class' : 'logistic regression per layer');
const scal = (X) => ((sel(X).scaling || 'weight') === 'weight' ? 'μ = σ(W)' : 'stretched 0.01–0.99');
const mono = (X) => (sel(X).monotone === false ? 'off' : 'on');
const ncls = (X) => sel(X).n_classes || 10;
const kap = (X) => sel(X).kappa || 0;
const comb = (X) => ((sel(X).combination || 'network') === 'network' ? 'inference network' : `fuzzy γ = ${g(sel(X).gamma || 0.9, 2)}`);
const other = (name) => (C.has(name) ? C.load(name) : null);
let tn = 3;
const T = () => ++tn;
const DEF = 'WofE, μ = σ(W), monotone, 10 classes, inference network (default)';
let memNo = 0;                 // number of the Carlin class table (referred to in Section 6.8)

const knowTable = (X, cap) => ['table', {
  caption: cap, head: ['Layer', 'Process', 'Direction / class scores', 'Expert weight'], widths: [2600, 1800, 3326, 1300],
  rows: X.knowledge.map((r) => [r.layer, r.process, r.direction === '+' ? 'high values favourable' : r.direction === '−' ? 'low values favourable' : r.direction, g(r.weight, 2)]),
}];
const layerTable = (X, cap) => ['table', {
  caption: cap, head: ['Layer', 'Calibration', 'Weight contrast', 'Trend z (expert direction)', 'Agrees', 'AUC of μ alone'],
  widths: [1700, 2726, 1100, 1300, 900, 1300],
  rows: tab(X).layers.map((r) => [r.layer, r.calibration, f(r.contrast, 2), f(r.trend_z, 1), r.agrees_with_expert ? 'yes' : '**no**', f(r.single_layer_auc)]),
}];
const memTable = (X, cap, layer) => ['table', {
  caption: cap, head: ['Class', 'Cells', 'Deposits', 'W ± s.d.', 'W used', 'μ data', 'μ expert'],
  widths: [2226, 1100, 1000, 1500, 1000, 1100, 1100],
  rows: tab(X).memberships.filter((r) => r.layer === layer).map((r) => [r.cls, int(r.cells), String(r.deposits),
    pm(r.weight, r.weight_sd, 2), f(r.weight_used, 2), f(r.mu_data, 2), f(r.mu_expert, 2)]),
}];
const best = (X) => tab(X).layers.slice().sort((a, b) => b.single_layer_auc - a.single_layer_auc)[0];
const worst = (X) => tab(X).layers.slice().sort((a, b) => a.single_layer_auc - b.single_layer_auc)[0];
const against = (X) => tab(X).layers.filter((r) => !r.agrees_with_expert).map((r) => r.layer);

const interp = (X, label) => {
  const t = [`The selected setting is **${sel(X).label}**.`];
  t.push(`The best calibrated layer is ${best(X).layer} (AUC of its membership alone ${f(best(X).single_layer_auc)}), the weakest ${worst(X).layer} (${f(worst(X).single_layer_auc)}) (Table ${label}).`);
  const a = against(X);
  t.push(a.length ? `${a.join(' and ')} ${a.length > 1 ? 'run' : 'runs'} significantly against the expert direction and ${a.length > 1 ? 'are' : 'is'} set neutral.` : 'No layer runs significantly against its expert direction.');
  return t.join(' ');
};

const setupTable = (cap) => ['table', {
  caption: cap, head: ['Dialog field', 'Carlin (Au)', 'Curnamona (REE)'], widths: [2626, 3200, 3200], rows: [
    ['Project (*Project ▸ Open demo project*)', 'Carlin-type Au, Nevada (USGS)', 'REE, Curnamona, South Australia'],
    ['Deposits (points) / study area mask', 'deposits / study_mask', 'deposits / study_mask'],
    ['Hold out for testing', '30 % spatially stratified (validation runs); 0 % for the final map', '0 % (leave-one-deposit-out in this document)'],
    ['Thin deposits / absence points', 'off / all background cells', 'off / all background cells'],
    ['Knowledge table (ticked rows)', J.knowledge.map((r) => `${r.layer} → ${r.process}, ${r.direction === '+' ? 'high' : r.direction === '−' ? 'low' : 'scores ' + r.direction}${r.direction.length < 2 ? ' values favourable' : ''}, weight ${g(r.weight, 2)}`).join('; '),
      K.knowledge.map((r) => `${r.layer} → ${r.process}, ${r.direction === '+' ? 'high' : r.direction === '−' ? 'low' : 'scores ' + r.direction}${r.direction.length < 2 ? ' values favourable' : ''}, weight ${g(r.weight, 2)}`).join('; ')],
    ['Process weights', 'all 1', 'all 1'],
    ['Membership calibration', cal(J), cal(K)],
    ['Membership scaling', scal(J), scal(K)],
    ['Keep the expert direction', mono(J), mono(K)],
    ['Classes per continuous layer', String(ncls(J)), String(ncls(K))],
    ['Expert worth κ', g(kap(J), 3), g(kap(K), 3)],
    ['Combination', comb(J), comb(K)],
    ['Output membership / process maps, prefix', 'off / off, dcfuzzy', 'off / off, dcfuzzy'],
    ['Result', `test AUC ${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)} (10 splits)`, `LODO AUC ${f(lodo(K))}`],
  ] }];

const compare = () => {
  const rows = [['**Data-calibrated fuzzy (this document)**', `**${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}**`, `**${f(lodo(K))}**`]];
  const add = (key, label) => {
    const a = other(`${key}_carlin`), b = other(`${key}_curnamona`);
    if (!a && !b) return;
    rows.push([label, a && a.robust_summary ? pm(a.robust_summary.test_auc_mean, a.robust_summary.test_auc_sd) : '—',
      b ? f(b.search.find((s) => s.label === b.selected.label).mean_test_auc) : '—']);
  };
  add('fuzzy_wofe', 'Fuzzy weights of evidence (hybrid, same knowledge table)');
  add('neuro_fuzzy', 'Neuro-fuzzy inference (hybrid, same knowledge table)');
  add('wofe', 'Weights of evidence (data-driven)');
  add('logistic_regression', 'Logistic regression (data-driven)');
  return ['table', { caption: `Table 6.${T()}. Data-calibrated fuzzy model compared with other MPM Studio methods (same validation protocol; data-driven methods use their own evidence sets).`,
    head: ['Method', 'Carlin: test AUC (10 splits)', 'Curnamona: mean LODO AUC'], widths: [4026, 2500, 2500], rows }];
};

module.exports = {
  file: 'dc_fuzzy.docx',
  code: 'DCF',
  short: 'Data-calibrated fuzzy memberships',
  title: 'Data-Calibrated Fuzzy Memberships (WofE / LR) on the Expert Inference Network',
  subtitle: 'Hybrid method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Fuzzy logic overlays combine evidence maps through memberships in the set of favourable locations. In a knowledge-driven fuzzy model the expert assigns the memberships; in a data-driven fuzzy model they are derived from the known deposits, while the way the memberships are combined — the inference network — still encodes the expert\'s conceptual model (Porwal et al., 2003). This hybrid splits the roles cleanly: the expert decides **which** evidence represents **which** mineral-system process, in **which** direction it is favourable, and **how** the processes combine; the deposits decide **how favourable** every evidence value is. MPM Studio calibrates the memberships either with multi-class weights of evidence or with a logistic regression per layer, keeps them monotone in the expert direction if asked, and reports every layer whose data contradict the expert. MPM Studio: *Hybrid ▸ Fuzzy evidence ▸ Data-Calibrated Fuzzy Memberships (WofE / LR)*.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4200, 4826], rows: [
      ['`python/logic/hybrid/dc_fuzzy/model.py`', 'Class weights, isotonic fit, 1-D logistic regression, membership scaling, expert pooling, inference network / γ operator, tables'],
      ['`python/logic/hybrid/dc_fuzzy/plugin.py`, `dialogs.py`', 'Registration, outputs, dialog'],
      ['`python/logic/hybrid/common/*`', 'Knowledge table, robust scaling, expert memberships, process scores and favourability shared by the hybrid methods'],
      ['`python/tests/test_dc_fuzzy.py`', 'Weights by hand, isotonic and logistic fits against scikit-learn, γ operator, direction check, pooling limits, Carlin model'],
    ] }],
    ['note', 'Sources of the implementation: data-derived memberships combined by a knowledge-based inference network follow the data-driven fuzzy model of Porwal et al. (2003); the class weights are the multi-class weights of evidence of Bonham-Carter (1994); the monotone fit is the pool-adjacent-violators algorithm (Ayer et al., 1955); the γ operator follows Zimmermann and Zysno (1980) and An et al. (1991). The logistic calibration per layer, the scaling μ = σ(W), the significance test of the expert direction and the pooling with the expert memberships are MPM Studio design choices (Sections 2.3–2.5).'],
    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Knowledge-driven, data-driven and hybrid fuzzy models'],
    ['p', 'A fuzzy membership μ ∈ [0, 1] expresses how strongly an evidence value supports the hypothesis "this cell is prospective". Knowledge-driven fuzzy models assign μ from experience — which makes them usable where deposits are unknown but subjective where they are known. Data-driven fuzzy models replace the expert memberships by values estimated from the deposits; Porwal et al. (2003) derived them from class-wise favourability and kept a knowledge-based inference network, and showed that the data-driven memberships outperformed the expert ones where enough deposits were available. The combination step remains a modelling decision of the expert: OR-like operators where proxies can replace each other (two pathfinder elements of one halo), AND-like operators where every ingredient of the mineral system is required (source, pathway and trap).'],
    ['h2', '2.2 Weights of evidence as memberships'],
    ['p', 'The weight of a class, the log-ratio of its frequency among deposit cells and among non-deposit cells, is the evidence the class provides in log-odds: W = 0 means no information, W > 0 favourable, W < 0 unfavourable (Bonham-Carter, 1994). The logistic function maps log-odds to a probability scale, so μ = σ(W) is a membership with a direct meaning: 0.5 is neutral, and a layer with weak weights stays close to 0.5 and influences the AND/OR combination little. Alternatively the weights of each layer can be stretched to the full range 0.01–0.99, as an expert would score every layer from "unfavourable" to "favourable" — which gives weak and strong layers the same leverage.'],
    ['h2', '2.3 Keeping the expert direction'],
    ['p', 'With few deposits the class weights of a continuous layer are noisy: one class with two deposits can outrank its neighbours by chance. The expert direction — closer to faults is better, more Sb is better — is prior knowledge about the **shape** of the response; a weighted isotonic regression (weights 1/s²(W)) finds the monotone sequence of weights closest to the observed ones, pooling adjacent classes that violate the order. The deposits still decide the size of every step. If the weights run significantly against the expert direction (weighted trend z < −1.96), the layer is reported: either the direction in the knowledge table is wrong or the deposits sample a different deposit type.'],
    ['h2', '2.4 Logistic calibration'],
    ['p', 'Classes discretise a continuous layer; a logistic regression of the deposit indicator on the robustly scaled values gives a smooth calibration with two parameters per layer — robust with few deposits — whose log-odds against the prior play the role of the class weight. Its slope b is the change of log-odds per inter-quartile range; b < 0 contradicts the expert direction and, when the direction is kept, is set to 0 (the layer becomes neutral). Categorical layers (e.g. a geology map) keep the class weights, which equal the class log-odds of a one-hot logistic regression.'],
    ['h2', '2.5 Pooling with the expert'],
    ['p', 'Where deposits are very few, the calibrated memberships can be pooled with the expert ones, μ = λ μ_data + (1 − λ) μ_expert with λ = N/(N + κ): κ is the worth of the expert knowledge in deposits — κ = N gives equal shares, κ → ∞ the expert fuzzy model.'],
    ['h1', '3 Mathematical basis'],
    ['p', 'Layer j with classes c (categorical classes or quantile classes of a continuous layer), $n_{c}$ cells and $d_{c}$ training deposit cells among the n cells and N deposit cells where the layer has data (an empty count is replaced by ½):'],
    ['eq', 'W_{c}=\\ln\\frac{d_{c}/N}{(n_{c}-d_{c})/(n-N)},      s^{2}(W_{c})=\\frac{1}{d_{c}}+\\frac{1}{n_{c}-d_{c}}', 'dc-w'],
    ['p', 'Monotone fit along the expert direction (classes in ascending value, increasing for d = +1, decreasing for d = −1):'],
    ['eq', '\\tilde{W}=\\rm{arg}\\min_{\\tilde{W}_{1}≤…≤\\tilde{W}_{k}}\\sum_{c}{\\frac{(W_{c}-\\tilde{W}_{c})^{2}}{s^{2}(W_{c})}}', 'dc-iso'],
    ['p', 'Direction test: weighted least-squares slope of $W_{c}$ on the class order with weights $1/s^{2}$, $z=d\\cdot\\hat{β}/\\rm{se}(\\hat{β})$; z < −1.96 is reported. Logistic calibration of a continuous layer on all cells with data:'],
    ['eq', '\\rm{logit}P(x)=a+b\\,u(x),   u=d\\frac{x-\\rm{med}}{\\rm{IQR}},      W(x)=a+b\\,u(x)-\\rm{logit}\\frac{N}{n}', 'dc-lr'],
    ['p', '(maximum likelihood by Newton–Raphson; b := 0 if b < 0 and the direction is kept). Memberships and pooling:'],
    ['eq', 'μ=σ(W)  or  μ=σ\(s\(W-\\frac{W_{min}+W_{max}}{2}\)\),  s=\\frac{2\\ln 99}{W_{max}-W_{min}};      μ^{*}=λμ+(1-λ)μ_{expert},  λ=\\frac{N}{N+κ}', 'dc-mu'],
    ['p', 'Combination by the inference network of the knowledge table — process score $F_{k}=(\\sum_{j∈k}{w_{j}μ_{j}^{3}})^{1/3}$ (weights renormalised over the observed proxies), favourability $\\prod_{k}{F_{k}^{v_{k}}}$ — or by the γ operator over the observed layers:'],
    ['eq', 'μ_{γ}=\(1-\\prod_{j}{(1-μ_{j})}\)^{γ}\(\\prod_{j}{μ_{j}}\)^{1-γ}', 'dc-gamma'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Knowledge network** — robust directed scaling and expert memberships of all study cells.',
      '**Classes** — categorical classes or quantile classes (default 10) of every layer.',
      '**Calibration** — class weights [[dc-w]] (monotone fit [[dc-iso]] if asked) or logistic regression [[dc-lr]] per continuous layer; direction test.',
      '**Memberships** [[dc-mu]] — scaling, optional pooling with the expert memberships; missing evidence stays missing.',
      '**Combination** — inference network or γ operator [[dc-gamma]].',
      '**Outputs** — favourability, expert map, limiting process, optional membership and process maps; layer and class tables; validation.',
    ]],
    ['h1', '5 How to use'],
    ['table', { caption: 'Table 5.1. Settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2300, 1900, 4826], rows: [
      ['Deposits / mask / hold-out / thinning', 'deposit layer, project mask, 20 %, off', 'As for the other hybrid methods; non-deposits are not sampled (every cell counts).'],
      ['Knowledge table', '—', 'Process, direction of continuous layers, expert class scores (used for the expert memberships when pooling and for the expert map), expert weights and process weights (used by the inference network).'],
      ['Membership calibration', 'WofE', 'WofE: class weights (any shape, monotone if asked). LR: a smooth monotone logistic function per continuous layer (two parameters, robust with few deposits).'],
      ['Membership scaling', 'μ = σ(W)', 'σ(W) keeps the evidence strength of each layer; stretched gives every layer the full range.'],
      ['Keep the expert direction', 'on', 'Monotone memberships; a layer against its expert direction is flattened and reported.'],
      ['Classes per continuous layer', '10', 'Fewer with few deposits.'],
      ['Expert worth κ', '0', 'Pooling with the expert memberships, λ = N/(N + κ).'],
      ['Combination / γ', 'inference network / 0.9', 'Network: OR within a process, AND between processes. γ: one operator over all layers.'],
    ] }],
    ['table', { caption: 'Table 5.2. Outputs.', head: ['Output', 'Content'], widths: [2300, 6726], rows: [
      ['`fuzzy`', 'Favourability from the data-calibrated memberships.'],
      ['`mu_<layer>` (optional)', 'Calibrated membership of every layer.'],
      ['`expert`, `limiting_process`, `process_*`', 'Expert favourability, weakest process and optional process scores.'],
      ['Tables', '`layers` (calibration, weight contrast, slope ± s.e., trend z, agrees with the expert, AUC of the membership alone), `memberships` (class, cells, deposits, W ± s.d., W used, μ data, μ expert, μ final), `summary`.'],
    ] }],
    ['p', 'Reading the results: the weight contrast is the range of the weights used — the strength of a layer; the AUC of a membership alone ranks the layers; a layer marked "no" contradicts its expert direction.'],
    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),
    ['h2', '6.5 Carlin (Au)'],
    knowTable(J, `Table 6.${T()}. Carlin — knowledge table.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (5 spatial 70 / 30 splits).`),
    C.robustTable(J, `Table 6.${T()}. Carlin — selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42).`),
    layerTable(J, `Table 6.${T()}. Carlin — calibrated layers (reference run).`),
    memTable(J, `Table 6.${(memNo = T())}. Carlin — classes of sb_log_idw.`, 'sb_log_idw'),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — data-calibrated fuzzy favourability with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) calibrated memberships per class (dashed: expert).'],
    ['p', `${interp(J, `6.${tn - 1}`)} Over 10 splits the selected setting reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}**. Default (σ(W), network) ${f(cand(J, DEF))}; LR stretched ${f(cand(J, 'LR, stretched'))}; monotone off: ${f(cand(J, 'WofE, μ = σ(W), not monotone'))}; γ operator: ${f(cand(J, 'WofE, μ = σ(W), monotone, fuzzy γ'))}.`],
    ['h2', '6.6 Curnamona (REE)'],
    knowTable(K, `Table 6.${T()}. Curnamona — knowledge table.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all occurrences.`),
    layerTable(K, `Table 6.${T()}. Curnamona — calibrated layers.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — data-calibrated fuzzy favourability with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) calibrated memberships.'],
    ['p', `${interp(K, `6.${tn}`)} Mean LODO AUC of the selected setting: **${f(lodo(K))}**; WofE ${f(cand(K, 'WofE, μ = σ(W), monotone, 10 classes'))}, LR ${f(cand(K, 'LR, μ = σ(W), monotone'))}, pooled with the expert ${f(cand(K, 'WofE, μ = σ(W), monotone, pooled'))}.`],
    compare(),
    ['h2', '6.7 Recommended setup for the demo projects'],
    ['p', 'The settings below reproduce the selected results from the dialog; Section 6.8 explains the reasoning.'],
    setupTable(`Table 6.${T()}. Data-calibrated fuzzy dialog settings for the demo projects.`),
    ['h2', '6.8 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal data-calibrated fuzzy settings.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1500, 1600, 1600, 4326],
      rows: [
        ['Evidence / processes', 'host (trap), faults (pathway), As + Sb (deposition)', 'Th + U (source), Nb (deposition), silica (footprint)', 'One to two proxies per mineral-system process: for Carlin-type gold the reactive host, the fault conduits and the As–Sb halo; for the REE system the U–Th–REE granite source, the co-enriched Nb and the quartz-rich cover that hides it.'],
        ['Calibration', cal(J), cal(K), `Class weights and the logistic calibration rank alike (Carlin WofE ${f(cand(J, 'WofE, stretched'))} / LR ${f(cand(J, 'LR, stretched'))} stretched; Curnamona ${f(cand(K, 'WofE, μ = σ(W), monotone, 10 classes'))} / ${f(cand(K, 'LR, μ = σ(W), monotone'))} with σ(W)); WofE is kept as the simpler, fully transparent choice (one weight per class, Table 6.${memNo}). The logistic function is the better choice for a layer whose response is smooth and where few deposits fall in each class.`],
        ['Scaling', scal(J), scal(K), `Carlin: stretching (${f(cand(J, 'WofE, stretched'))} vs ${f(cand(J, DEF))} for σ(W)) gives each proxy the full range. With σ(W) the host rock — by far the largest weight contrast (${f(tab(J).layers.find((r) => r.layer === 'geology_host').contrast, 1)}) — would dominate the AND-like combination, while the mineral system needs the fault pathway and the As–Sb footprint as much as the trap. Curnamona: σ(W) is better (${f(cand(K, DEF))} vs ${f(cand(K, 'WofE, stretched'))}) because Nb and the silica index carry little contrast (${f(tab(K).layers.find((r) => r.layer === 'geochem_nb_log_idw').contrast, 1)}, ${f(tab(K).layers.find((r) => r.layer === 'aster_silica_index').contrast, 1)}) at seven occurrences; stretching them would inflate noise to the same leverage as the strong radiometric U and Th.`],
        ['Expert direction kept', mono(J), mono(K), `Essential: without the monotone fit the AUC falls to ${f(cand(J, 'WofE, μ = σ(W), not monotone'))} (Carlin) and ${f(cand(K, 'WofE, μ = σ(W), not monotone'))} (Curnamona). With a few deposits per class the raw class weights jump up and down by chance; the expert direction (closer to faults, richer in Sb, more radiometric U) removes those jumps while the deposits still set the steps. All layers follow their expert direction (trend z ${f(tab(J).layers.find((r) => r.layer === 'sb_log_idw').trend_z, 1)} for Sb on Carlin, ${f(tab(K).layers.find((r) => r.layer === 'rad_u').trend_z, 1)} for U on Curnamona).`],
        ['Classes', String(ncls(J)), String(ncls(K)), `Ten classes on both projects (Carlin 5 / 20 classes: ${f(cand(J, 'WofE, μ = σ(W), monotone, 5 classes'))} / ${f(cand(J, 'WofE, μ = σ(W), monotone, 20 classes'))}; Curnamona ${f(cand(K, 'WofE, μ = σ(W), monotone, 5 classes'))} / ${f(cand(K, 'WofE, μ = σ(W), monotone, 20 classes'))}): five classes are too coarse to follow a dispersion halo or a distance gradient, twenty leave most classes without a deposit.`],
        ['Expert worth κ', g(kap(J), 3), g(kap(K), 3), `0 on both. Pooling with the expert memberships (κ = 20) helps the σ(W) variant (Carlin ${f(cand(J, 'WofE, μ = σ(W), monotone, pooled'))} vs ${f(cand(J, DEF))}; Curnamona ${f(cand(K, 'WofE, μ = σ(W), monotone, pooled'))} vs ${f(cand(K, DEF))}) but not beyond the selected settings — the monotone constraint already carries the expert knowledge the data need.`],
        ['Combination', comb(J), comb(K), `Carlin: the AND-like inference network (${f(cand(J, DEF))}) beats γ = 0.9 (${f(cand(J, 'WofE, μ = σ(W), monotone, fuzzy γ'))}): a Carlin-type deposit needs a reactive host **and** a fault conduit **and** an As–Sb footprint, and the OR-leaning γ lets one strong proxy compensate for a missing ingredient. Curnamona: γ = 0.9 wins (${f(cand(K, 'WofE, μ = σ(W), monotone, fuzzy γ'))} vs ${f(cand(K, DEF))}): the occurrences are flagged by any strong radiometric or Nb signal, whereas the AND-like network penalises occurrences where the weak cover proxy (silica index, AUC ${f(tab(K).layers.find((r) => r.layer === 'aster_silica_index').single_layer_auc)}) is unfavourable.`],
      ],
    }],
    ['h2', '6.9 Discussion'],
    ['bullets', [
      `**Accuracy.** Carlin test AUC ${f(J.robust_summary.test_auc_mean)} (10 splits); Curnamona mean LODO AUC ${f(lodo(K))}.`,
      '**Division of labour.** The expert supplies what deposits cannot reveal with tens of examples — the structure of the mineral system and the direction of every proxy; the deposits supply what experts estimate worst — where along each proxy favourability rises and how strongly.',
      '**Transparency.** Every membership is a class weight or a two-parameter logistic function with its standard error; a contradiction between data and knowledge is shown, not hidden.',
      '**Limits.** The class weights of continuous layers need several deposits per class; the inference network still assumes that proxies of one process substitute and processes multiply — correlated proxies across processes are counted twice (fuzzy WofE\'s modified model estimates such dependence instead).',
    ]],
    ['h1', '7 References'],
    ['refs', R('an1991', 'ayer1955', 'bonham1994', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012',
      'fawcett2006', 'hofstra2000', 'muntean2011', 'neumann2000', 'porwal2003b', 'raines2001', 'roberts2017',
      'stevens2004', 'yousefi2015', 'zimmermann1980')],
  ],
};
