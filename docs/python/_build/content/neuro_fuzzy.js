const { R } = require('../refs');
const C = require('../common');

const { f, g, pm } = C;
const J = C.load('neuro_fuzzy_carlin');
const K = C.load('neuro_fuzzy_curnamona');
const tab = (X) => X.reference.tables;
const st = (X) => tab(X).stats;
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const cand = (X, start) => { const r = X.search.find((s) => s.label.startsWith(start)); return r ? r.mean_test_auc : null; };
const DEF = 'one input per layer, 2 MFs, first order, κ = 1, checking 25 % (default)';
const sel = (X) => X.selected;
const kappa = (X) => (sel(X).kappa === undefined ? 1 : sel(X).kappa);
const nmf = (X) => sel(X).n_mf || 2;
const cons = (X) => ((sel(X).consequent || 'linear') === 'linear' ? 'first order (p·x + q)' : 'zero order (q)');
const inputs = (X) => (tab(X).input_mode === 'processes' ? 'one per process' : 'one per layer');
const lr = (X) => (sel(X).lr === undefined ? 0.02 : sel(X).lr);
const chk = (X) => (sel(X).check_pct === undefined ? 25 : sel(X).check_pct);
const other = (name) => (C.has(name) ? C.load(name) : null);
let tn = 3;
const T = () => ++tn;
let ruleNoJ = 0;               // number of the Carlin rule table (referred to in Section 6.8)

const knowTable = (X, cap) => ['table', {
  caption: cap, head: ['Layer', 'Process', 'Direction / class scores', 'Expert weight'], widths: [2600, 1800, 3326, 1300],
  rows: X.knowledge.map((r) => [r.layer, r.process, r.direction === '+' ? 'high values favourable' : r.direction === '−' ? 'low values favourable' : r.direction, g(r.weight, 2)]),
}];
const mfTable = (X, cap) => ['table', {
  caption: cap, head: ['Input', 'MF', 'Centre c: expert → learnt (± s.d. over models)', 'Width σ: expert → learnt'],
  widths: [2400, 1100, 3326, 2200],
  rows: tab(X).membership.map((r) => [r.input, r.mf, `${f(r.centre_expert, 2)} → ${pm(r.centre, r.centre_sd, 2)}`, `${f(r.sigma_expert, 3)} → ${f(r.sigma, 3)}`]),
}];
const ruleTable = (X, cap, n) => ['table', {
  caption: cap, head: ['Rule (IF …)', 'Expert K', 'Learnt output', 'Area %', 'Deposits %', 'Lift'],
  widths: [4026, 950, 1050, 950, 1050, 1000],
  rows: tab(X).rules.filter((r) => r.firing_area_pct >= 0.1).slice(0, n).map((r) => [r.antecedent.replace(/ AND /g, ' ∧ '), f(r.expert_K, 2), f(r.output, 2),
    f(r.firing_area_pct, 1), f(r.firing_deposits_pct, 1), f(r.lift, 2)]),
}];
const impTable = (X, cap) => ['table', {
  caption: cap, head: ['Input', 'AUC drop when permuted (mean ± s.d. over models)'], widths: [4026, 5000],
  rows: tab(X).importance.map((r) => [r.input, pm(r.auc_drop, r.auc_drop_sd, 3)]),
}];
// the mean learnt "high" centre and "low" centre of an input
const mfOf = (X, inp, mf) => tab(X).membership.find((r) => r.input === inp && r.mf === mf);
const topRule = (X) => tab(X).rules.filter((r) => r.firing_area_pct >= 0.1)[0];
const lowest = (X) => { const a = tab(X).rules.filter((r) => r.firing_area_pct >= 0.1); return a[a.length - 1]; };
const topImp = (X) => tab(X).importance[0];
const lastImp = (X) => tab(X).importance[tab(X).importance.length - 1];

const interp = (X, label) => {
  const t = [`The selected setting is **${sel(X).label}** (${st(X).inputs} inputs, ${st(X).rules} rules).`];
  const tr = topRule(X), lo = lowest(X);
  t.push(`The strongest rule is *IF ${tr.antecedent.replace(/ AND /g, ' and ')}* (learnt output ${f(tr.output, 2)}, expert start ${f(tr.expert_K, 2)}); it covers ${f(tr.firing_area_pct, 1)} % of the study area but ${f(tr.firing_deposits_pct, 1)} % of the training deposits (lift ${f(tr.lift, 1)}). The weakest is *IF ${lo.antecedent.replace(/ AND /g, ' and ')}* (output ${f(lo.output, 2)}).`);
  t.push(`Permutation importance ranks ${topImp(X).input} first (AUC drop ${f(topImp(X).auc_drop, 3)}) and ${lastImp(X).input} last (${f(lastImp(X).auc_drop, 3)}) (Table ${label}).`);
  return t.join(' ');
};

const setupTable = (cap) => ['table', {
  caption: cap, head: ['Dialog field', 'Carlin (Au)', 'Curnamona (REE)'], widths: [2626, 3200, 3200], rows: [
    ['Project (*Project ▸ Open demo project*)', 'Carlin-type Au, Nevada (USGS)', 'REE, Curnamona, South Australia'],
    ['Deposits (points) / study area mask', 'deposits / study_mask', 'deposits / study_mask'],
    ['Hold out for testing', '30 % spatially stratified (validation runs); 0 % for the final map', '0 % (leave-one-deposit-out in this document)'],
    ['Thin deposits / min. distance of non-deposits / absence points', 'off / 3 cells (3 km) / all background cells', 'off / 3 cells / all background cells'],
    ['Knowledge table (ticked rows)', J.knowledge.map((r) => `${r.layer} → ${r.process}, ${r.direction === '+' ? 'high' : r.direction === '−' ? 'low' : 'scores ' + r.direction}${r.direction.length < 2 ? ' values favourable' : ''}, weight ${g(r.weight, 2)}`).join('; '),
      K.knowledge.map((r) => `${r.layer} → ${r.process}, ${r.direction === '+' ? 'high' : r.direction === '−' ? 'low' : 'scores ' + r.direction}${r.direction.length < 2 ? ' values favourable' : ''}, weight ${g(r.weight, 2)}`).join('; ')],
    ['Process weights', 'all 1', 'all 1'],
    ['Inputs', sel(J).inputs ? inputs(J) : 'auto (= one per layer, 16 rules)', sel(K).inputs ? inputs(K) : 'auto (= one per layer, 16 rules)'],
    ['Membership functions per input', String(nmf(J)), String(nmf(K))],
    ['Rule outputs', cons(J), cons(K)],
    ['Knowledge prior κ', g(kappa(J), 3), g(kappa(K), 3)],
    ['Epochs / learning rate (premises)', `100 / ${g(lr(J), 2)}`, `100 / ${g(lr(K), 2)}`],
    ['Checking set', chk(J) ? `${chk(J)} %` : 'off', chk(K) ? `${chk(K)} %` : 'off'],
    ['Non-deposits per deposit / models (bags)', '3 / 10', '3 / 10'],
    ['Output process maps / prefix', 'off / neurofuzzy', 'off / neurofuzzy'],
    ['Result', `test AUC ${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)} (10 splits)`, `LODO AUC ${f(lodo(K))}`],
  ] }];

const compare = () => {
  const rows = [];
  const add = (name, key, label) => {
    const a = other(`${key}_carlin`), b = other(`${key}_curnamona`);
    if (!a && !b) return;
    rows.push([label, a && a.robust_summary ? pm(a.robust_summary.test_auc_mean, a.robust_summary.test_auc_sd) : '—',
      b ? f(b.search.find((s) => s.label === b.selected.label).mean_test_auc) : '—']);
  };
  rows.push(['**Neuro-fuzzy (this document)**', `**${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}**`, `**${f(lodo(K))}**`]);
  add('', 'fuzzy_wofe', 'Fuzzy weights of evidence (hybrid)');
  add('', 'mlp', 'Neural network (MLP, data-driven)');
  add('', 'random_forest', 'Random forest (data-driven)');
  add('', 'logistic_regression', 'Logistic regression (data-driven)');
  return ['table', { caption: `Table 6.${T()}. Neuro-fuzzy model compared with other MPM Studio methods (same protocol; the data-driven methods use their own evidence sets, see their documents).`,
    head: ['Method', 'Carlin: test AUC (10 splits)', 'Curnamona: mean LODO AUC'], widths: [4026, 2500, 2500], rows }];
};

module.exports = {
  file: 'neuro_fuzzy.docx',
  code: 'NF',
  short: 'Neuro-fuzzy inference (ANFIS-type)',
  title: 'Neuro-Fuzzy Inference (ANFIS-type) on the Mineral-System Knowledge Network',
  subtitle: 'Hybrid method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'A neuro-fuzzy system is a fuzzy inference system whose membership functions and rule outputs are adjusted by the learning algorithms of neural networks. Its rules keep the linguistic form of expert reasoning — *IF the host rock is favourable AND the cell is close to a fault THEN the prospectivity is high* — while the deposits decide where "favourable" and "close" begin and how high each rule scores. Porwal et al. (2004) introduced the approach to mineral potential mapping: a Takagi–Sugeno fuzzy inference system (Takagi and Sugeno, 1985) with knowledge-based membership functions, whose parameters are estimated by the adaptive-network training of ANFIS (Jang, 1993). MPM Studio builds the rule base on its mineral-system knowledge network (shared by all hybrid tools), starts every membership function and every rule from the expert model, and lets the hybrid learning rule correct them, shrinking rules that the data cannot inform back to the expert answer. MPM Studio: *Hybrid ▸ Neuro-fuzzy ▸ Neuro-Fuzzy Inference (ANFIS-type)*.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4200, 4826], rows: [
      ['`python/logic/hybrid/neuro_fuzzy/model.py`', 'Takagi–Sugeno system, hybrid learning (least squares + Adam), knowledge prior, checking-set early stopping, bagging, rule / membership / importance tables'],
      ['`python/logic/hybrid/neuro_fuzzy/plugin.py`, `dialogs.py`', 'Registration, outputs, dialog'],
      ['`python/logic/hybrid/common/*`', 'Knowledge table, robust scaling, expert memberships, process scores and favourability shared by the hybrid methods'],
      ['`python/tests/test_neuro_fuzzy.py`', 'Forward pass = definition, premise gradient = finite differences, prior limits, rule recovery, input choice, Carlin model'],
    ] }],
    ['note', 'Sources of the implementation: the first-order Takagi–Sugeno rule base [[nf-w]]–[[nf-y]] follows Takagi and Sugeno (1985); the hybrid learning rule — least squares for the consequents in the forward pass, gradient descent for the premise parameters in the backward pass — and the checking-set early stopping follow Jang (1993); knowledge-based membership functions with data-estimated parameters follow Porwal et al. (2004). The premise step uses Adam (Kingma and Ba, 2015) instead of Jang\'s step-size heuristics, and the knowledge prior on the consequents [[nf-lse]] is an MPM Studio design choice that ties the method to the expert model of the other hybrid tools (Section 2.4).'],
    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Fuzzy inference with Takagi–Sugeno rules'],
    ['p', 'In fuzzy set theory an element belongs to a set to a degree μ ∈ [0, 1]. A fuzzy inference system maps inputs to an output through IF–THEN rules whose antecedents combine fuzzy sets ("x₁ is high AND x₂ is low"). In the Takagi–Sugeno (TS) form the consequent of every rule is a function of the inputs, usually linear; the output is the average of the rule consequents weighted by the degree to which each antecedent is satisfied (the firing strength). TS systems are smooth, need no defuzzification and are universal approximators, yet each rule can still be read as a statement about the deposit model.'],
    ['h2', '2.2 ANFIS: adaptive networks and hybrid learning'],
    ['p', 'Jang (1993) wrote a TS system as a five-layer feed-forward network — membership functions, rule firing (product), normalisation, rule consequents, sum — so that its parameters can be learnt from examples. The output is linear in the consequent parameters once the premise (membership) parameters are fixed, which gives the **hybrid learning rule**: in each epoch the consequents are the least-squares solution for the current premises, and the premises take one gradient step on the squared error. Least squares makes learning much faster than back-propagation alone and avoids its local minima in the linear part. To limit over-fitting Jang monitored the error on a separate **checking set** and kept the parameters of the epoch with the smallest checking error.'],
    ['h2', '2.3 Neuro-fuzzy models for mineral potential mapping'],
    ['p', 'Porwal et al. (2004) used a TS system whose inputs are evidence maps already transformed by expert knowledge into favourability values, with Gaussian membership functions whose initial parameters came from the expert and whose final parameters were estimated by ANFIS on deposit and non-deposit locations. Two problems follow from the size of mineral exploration data. First, a grid-partition rule base has M^n rules for n inputs and M membership functions each, so the inputs must be few. Second, with tens of deposits many rules cover no deposit at all and their outputs are undetermined. MPM Studio answers the first with the knowledge network: the inputs are the expert memberships of the evidence layers (one per layer; few layers) or the expert scores of the mineral-system processes (one per process; many layers). It answers the second with a knowledge prior on the rule outputs.'],
    ['h2', '2.4 The knowledge prior on the rules'],
    ['p', 'Every rule has a prototype — the input vector at the centres of its membership functions — and the expert network gives a favourability K at that prototype: the rule *IF all inputs are high* starts from K ≈ 1, the rule *IF all inputs are low* from K ≈ 0, mixed rules in between according to the AND-like combination of the processes. These expert values, calibrated to the sample by a logistic (Platt) fit of logit K, are the prior means of the rule constants; the least-squares step becomes a ridge regression that shrinks every constant towards its prior and every slope towards 0 with weight κ. A rule fired by many training samples follows the data; a rule the data never reach keeps the expert answer instead of an arbitrary value. κ = 0 is a purely data-driven ANFIS, large κ the fixed expert rule base.'],
    ['h2', '2.5 Over-fitting control and calibration'],
    ['p', 'Three devices keep the model stable with few deposits: (1) the checking set — a stratified 25 % of the samples held out to pick the epoch with the smallest checking error; (2) bagging (Breiman, 1996) — ten models, each with its own non-deposit draw and checking split, averaged; (3) Platt calibration (Platt, 1999) of each model\'s output on its sample, which turns the regression output into a probability-like score without changing the ranking of the cells.'],
    ['h1', '3 Mathematical basis'],
    ['p', 'Inputs $x_{i}∈[0,1]$, $i=1…n$: expert process scores $F_{k}$ or expert memberships $μ_{j}$ (missing = 0.5), directed so that 1 is favourable. Each input has M Gaussian membership functions ("low", ("medium",) "high"):'],
    ['eq', 'A_{im}(x)=\\exp\(-\\frac{(x-c_{im})^{2}}{2σ_{im}^{2}}\),      c^{0}_{im}=\\frac{m-1}{M-1},  σ^{0}=\\frac{1/(M-1)}{2\\sqrt{2\\ln 2}}', 'nf-mf'],
    ['p', 'so that neighbouring expert membership functions cross at 0.5. The grid partition gives $R=M^{n}$ rules $r$, each choosing one function $m_{r,i}$ per input:'],
    ['eq', 'w_{r}=\\prod_{i}{A_{i,m_{r,i}}(x_{i})},      \\bar{w}_{r}=\\frac{w_{r}}{\\sum_{s}{w_{s}}}', 'nf-w'],
    ['eq', 'f_{r}(x)=p_{r}^{T}x+q_{r},      \\hat{y}(x)=\\sum_{r}{\\bar{w}_{r}f_{r}(x)}', 'nf-y'],
    ['p', '(zero-order rules: $f_{r}=q_{r}$). Firing strengths are computed in log space and normalised with a soft-max. With samples $(x_{t},y_{t})$, $y=1$ for training deposit cells and 0 for non-deposit cells, weights $s_{t}=1$, and the prior means $q^{0}_{r}=σ(a\\,\\rm{logit}K(\\tilde{x}_{r})+b)$ at the rule prototypes $\\tilde{x}_{r}$ (a, b from a Platt fit of logit K on the sample), the consequents $θ=(p,q)$ solve'],
    ['eq', '\\min_{θ}\\sum_{t}{s_{t}(y_{t}-\\hat{y}_{t})^{2}}+κ\\sum_{r}{[(q_{r}-q^{0}_{r})^{2}+\\|p_{r}\\|^{2}]}  ⇔  (Φ^{T}SΦ+κI)θ=Φ^{T}Sy+κθ^{0}', 'nf-lse'],
    ['p', 'with the design matrix $Φ_{t}=[\\bar{w}_{t1}x_{t},\\bar{w}_{t1},…,\\bar{w}_{tR}x_{t},\\bar{w}_{tR}]$. Gradient of the weighted mean squared error $E$ for the premises (the consequents are at their optimum, so their dependence on the premises drops out):'],
    ['eq', '\\frac{∂E}{∂c_{im}}=\\sum_{t}{h_{tim}\\frac{x_{ti}-c_{im}}{σ_{im}^{2}}},   \\frac{∂E}{∂\\lnσ_{im}}=\\sum_{t}{h_{tim}\\frac{(x_{ti}-c_{im})^{2}}{σ_{im}^{2}}},   h_{tim}=\\frac{2s_{t}e_{t}}{\\sum s}\\sum_{r:m_{r,i}=m}{\\bar{w}_{tr}(f_{r}(x_{t})-\\hat{y}_{t})}', 'nf-grad'],
    ['p', 'with $e_{t}=\\hat{y}_{t}-y_{t}$. Adam updates $c$ and $\\lnσ$ (centres kept in [−0.5, 1.5], $σ∈[0.03,2]$). After training, $P=σ(α\\hat{y}+β)$ with α, β from a Platt fit on the sample; the map is the mean of P over the B bagged models and `prob_sd` its standard deviation. Permutation importance of input i: the drop of the sample AUC when column i is permuted.'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Knowledge network** — robust directed scaling, expert memberships μ, process scores F and favourability K of all study cells (no deposits used).',
      '**Inputs** — μ of every layer or F of every process (*auto*: layers while M^m ≤ 64 rules); expert prototypes and their K.',
      '**For each of B models:** draw 3 non-deposit cells per training deposit (outside the deposit buffer); split off a stratified checking set; Platt-calibrate K on the training part to get the rule priors $q^{0}$.',
      '**Hybrid learning** for up to the given number of epochs: ridge least squares [[nf-lse]] for the consequents, one Adam step [[nf-grad]] for the premises; keep the epoch with the smallest checking error.',
      '**Calibrate and predict** — Platt fit of the model output on its sample; prediction of all study cells in chunks; firing shares of the rules over the area and the training deposits; permutation importance.',
      '**Combine** — mean and s.d. of P over the models; rule, membership and importance tables averaged over the models; validation of the mean map.',
    ]],
    ['h1', '5 How to use'],
    ['table', { caption: 'Table 5.1. Settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2300, 1900, 4826], rows: [
      ['Deposits / mask / hold-out / thinning / min. distance', 'deposit layer, project mask, 20 %, off, 3 cells', 'As for the other hybrid methods; non-deposit samples keep the minimum distance from every known deposit.'],
      ['Knowledge table', '—', 'Process, direction of continuous layers, expert score 0–1 per class of categorical layers, expert weights (used by the expert favourability K that sets the rule priors and, for process inputs, by the process scores).'],
      ['Inputs', 'auto', 'One per layer: the data shape every layer (rules grow as 2^m). One per process: few inputs and rules — required with many layers. Auto chooses layers while the rule base has ≤ 64 rules.'],
      ['Membership functions per input', '2 (low, high)', '3 adds a "medium" set and allows non-monotone responses, but multiplies the rules (3^n).'],
      ['Rule outputs', 'first order', 'First order f = p·x + q (Takagi–Sugeno, ANFIS); zero order f = q has fewer parameters.'],
      ['Knowledge prior κ', '1', 'Weight of the expert rule outputs in the least squares; 0 = data only, large = expert rule base kept.'],
      ['Epochs / learning rate', '100 / 0.02', 'Upper limit of the hybrid learning; learning rate 0 keeps the expert membership functions.'],
      ['Checking set', '25 %', 'Samples held out to pick the best epoch; off = last epoch.'],
      ['Non-deposits per deposit / models', '3 / 10', 'Sample size and bagging (more models = smoother map).'],
    ] }],
    ['table', { caption: 'Table 5.2. Outputs.', head: ['Output', 'Content'], widths: [2300, 6726], rows: [
      ['`prob`', 'Platt-calibrated neuro-fuzzy score, mean over the models.'],
      ['`prob_sd`', 'Standard deviation of the score over the models (sampling uncertainty).'],
      ['`expert`, `limiting_process`, `process_*`', 'Expert favourability, weakest process and optional process maps of the knowledge network.'],
      ['Tables', '`rules` (antecedent, expert K, learnt output at the prototype, constant q, firing share of the area and of the training deposits, lift), `membership` (expert and learnt c, σ), `importance` (permutation AUC drop), `bags` (samples, best epoch, errors, Platt, sample AUC), `summary`.'],
    ] }],
    ['p', 'Reading the results: a rule with lift > 1 covers more of the deposits than of the area; a learnt output far above its expert start shows a combination the expert underrated. A "high" membership function whose centre moves below 1 means the deposits are already favourable at intermediate expert memberships (the expert threshold was too strict); a centre above 1 or a narrow width, that only the most extreme values count.'],
    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),
    ['h2', '6.5 Carlin (Au)'],
    knowTable(J, `Table 6.${T()}. Carlin — knowledge table.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (5 spatial 70 / 30 splits).`),
    C.robustTable(J, `Table 6.${T()}. Carlin — selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42).`),
    mfTable(J, `Table 6.${T()}. Carlin — membership functions, expert start and learnt (reference run).`),
    ruleTable(J, `Table 6.${(ruleNoJ = T())}. Carlin — the eight strongest rules that fire on at least 0.1 % of the study area (reference run). Inputs are expert memberships: “high” = favourable in the expert direction (close to faults, low silica index).`, 8),
    impTable(J, `Table 6.${T()}. Carlin — permutation importance.`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — neuro-fuzzy calibrated score with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) learnt membership functions (dashed: expert start).'],
    ['p', `${interp(J, `6.${tn}`)} Over 10 splits the selected setting reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}**. Variants of the two-MF default (${f(cand(J, DEF))}): data only (κ = 0): ${f(cand(J, 'one input per layer, 2 MFs, first order, κ = 0'))}; strong prior (κ = 10): ${f(cand(J, 'one input per layer, 2 MFs, first order, κ = 10'))}; expert membership functions kept: ${f(cand(J, 'one input per layer, expert MFs'))}; process inputs: ${f(cand(J, 'one input per process, 2 MFs'))}.`],
    ['h2', '6.6 Curnamona (REE)'],
    knowTable(K, `Table 6.${T()}. Curnamona — knowledge table.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all occurrences.`),
    mfTable(K, `Table 6.${T()}. Curnamona — membership functions (reference run).`),
    ruleTable(K, `Table 6.${T()}. Curnamona — the eight strongest rules that fire on at least 0.1 % of the study area (reference run). Inputs are expert memberships: “high” = favourable in the expert direction (close to faults, low silica index).`, 8),
    impTable(K, `Table 6.${T()}. Curnamona — permutation importance.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — neuro-fuzzy calibrated score with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) learnt membership functions.'],
    ['p', `${interp(K, `6.${tn}`)} Mean LODO AUC of the selected setting: **${f(lodo(K))}**; with κ = 1 ${f(cand(K, DEF))}, data only (κ = 0) ${f(cand(K, 'one input per layer, 2 MFs, first order, κ = 0'))}. Inputs and rules read in the expert direction: “aster_silica_index is high” means a high membership, i.e. a low silica index (thin cover).`],
    compare(),
    ['h2', '6.7 Recommended setup for the demo projects'],
    ['p', 'The settings below reproduce the selected results from the neuro-fuzzy dialog; Section 6.8 explains the reasoning.'],
    setupTable(`Table 6.${T()}. Neuro-fuzzy dialog settings for the demo projects.`),
    ['h2', '6.8 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal neuro-fuzzy settings.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1500, 1600, 1600, 4326],
      rows: [
        ['Evidence / processes', 'host (trap), faults (pathway), As + Sb (deposition)', 'Th + U (source), Nb (deposition), silica (footprint)', 'One to two proxies per mineral-system process: for Carlin-type gold the reactive host, the fault conduits and the As–Sb halo; for the REE system the U–Th–REE granite source, the co-enriched Nb and the quartz-rich cover that hides it.'],
        ['Inputs', inputs(J), inputs(K), `Four layers give 2⁴ = 16 or 3⁴ = 81 rules — few enough for one input per layer, so every proxy keeps its own membership functions and the rules stay readable per proxy. Carlin: per layer ${f(cand(J, DEF))} versus per process ${f(cand(J, 'one input per process, 2 MFs'))} (2 MFs) and ${f(cand(J, 'one input per layer, 3 MFs'))} versus ${f(cand(J, 'one input per process, 3 MFs'))} (3 MFs) — nearly equal. Curnamona at κ = 1: per process ${f(cand(K, 'one input per process, 2 MFs'))} versus per layer ${f(cand(K, DEF))} — averaging Th and U, two proxies of one granite source, damps the noise of six occurrences — but the per-layer model with a stronger prior is better still (${f(cand(K, 'one input per layer, 2 MFs, first order, κ = 10'))}). Process inputs are the choice for many layers (2^m > 64 rules).`],
        ['Membership functions', String(nmf(J)), String(nmf(K)), `Carlin (${st(J).deposits} training deposits per split): a "medium" set lets the response to fault distance be non-monotone — the strongest rule needs a favourable host, high Sb and an intermediate fault proximity (Table 6.${ruleNoJ}), while the "very close to a fault" set (centre ${f(mfOf(J, 'dist_faults', 'high').centre, 2)}) is almost never fired by a deposit. Carlin deposits lie within a few kilometres of the mapped high-angle faults, not on their traces: the fluids spread laterally from the conduits into reactive, permeable carbonate beds, and the mapped faults are only the largest of the conduits (Cline et al., 2005; Muntean et al., 2011). Curnamona: seven occurrences cannot fill 81 rules (3 MFs ${f(cand(K, 'one input per layer, 3 MFs'))}); two sets suffice.`],
        ['Rule outputs', cons(J), cons(K), `First-order rules let the prospectivity rise inside the region of one rule (e.g. along the As–Sb dispersion gradient within the host); the knowledge prior shrinks the slopes to 0 where the samples are thin. Zero order: Carlin ${f(cand(J, 'one input per layer, 2 MFs, zero order'))}, Curnamona ${f(cand(K, 'one input per layer, 2 MFs, zero order'))}.`],
        ['Knowledge prior κ', g(kappa(J), 3), g(kappa(K), 3), `The decisive parameter. Without the prior (κ = 0: Carlin ${f(cand(J, 'one input per layer, 2 MFs, first order, κ = 0'))}, Curnamona ${f(cand(K, 'one input per layer, 2 MFs, first order, κ = 0'))}) every rule that contains no deposit gets an arbitrary least-squares output — e.g. "host unfavourable but high Sb" — and such combinations dominate the map outside the sampled area. Carlin's ${st(J).deposits} deposits inform the main rules, so κ = 1 lets the data move them; Curnamona's six training occurrences per fold need the expert rule base to carry most of the weight (κ = 1 ${f(cand(K, DEF))}, κ = 10 ${f(cand(K, 'one input per layer, 2 MFs, first order, κ = 10'))}). The prior is the geological model: unseen combinations behave as the mineral-system concept predicts.`],
        ['Membership learning', lr(J) ? `learnt (rate ${g(lr(J), 2)})` : 'expert kept', lr(K) ? `learnt (rate ${g(lr(K), 2)})` : 'expert kept', `Learning moves where "favourable" begins. Carlin: the high sets of Sb and As stay near 1 (${f(mfOf(J, 'sb_log_idw', 'high').centre, 2)}, ${f(mfOf(J, 'as_log_idw', 'high').centre, 2)}), the medium sets of the pathfinders settle at ${f(mfOf(J, 'sb_log_idw', 'medium').centre, 2)} (Sb) and ${f(mfOf(J, 'as_log_idw', 'medium').centre, 2)} (As) — moderate anomalies already count, as expected for a dispersion halo around the deposits. Curnamona: the high sets of Th and U move above 1 (${f(mfOf(K, 'rad_th', 'high').centre, 2)}, ${f(mfOf(K, 'rad_u', 'high').centre, 2)}) — only the strongest radiometric highs mark the U–Th–REE-rich granite — while Nb and the low-silica set move to ${f(mfOf(K, 'geochem_nb_log_idw', 'high').centre, 2)} and ${f(mfOf(K, 'aster_silica_index', 'high').centre, 2)}: intermediate Nb and thin cover are already favourable. Expert MFs kept: Carlin ${f(cand(J, 'one input per layer, expert MFs'))}, Curnamona ${f(cand(K, 'one input per layer, expert MFs'))}.`],
        ['Checking set', chk(J) ? `${chk(J)} %` : 'off', chk(K) ? `${chk(K)} %` : 'off', `The epoch with the smallest checking error (mean ${f(st(J).best_epoch, 0)} on Carlin, ${f(st(K).best_epoch, 0)} on Curnamona, of 100) stops the membership functions from wrapping around individual deposits. Without the checking set: Carlin ${f(cand(J, 'one input per layer, no checking'))}, Curnamona ${f(cand(K, 'one input per layer, no checking'))} — over-fitting costs most where deposits are fewest.`],
      ],
    }],
    ['h2', '6.9 Discussion'],
    ['bullets', [
      `**Accuracy.** Carlin test AUC ${f(J.robust_summary.test_auc_mean)} (10 splits); Curnamona mean LODO AUC ${f(lodo(K))}.`,
      '**Interpretability.** Every rule is an explicit statement about the deposit model, with its expert start, its learnt value and the share of area and deposits it covers — the model can be audited rule by rule, which a neural network or a random forest does not allow.',
      '**Knowledge versus data.** The expert decides the inputs, their directions and the starting rule base; the deposits move the membership thresholds and the rule outputs; κ sets how far.',
      '**Limits.** The rule base grows exponentially with the inputs (one input per process for many layers); with few deposits the membership functions move little and the model stays close to the expert network; a wrong expert direction is only partly corrected, because the rule priors and the input scaling both follow it.',
    ]],
    ['h1', '7 References'],
    ['refs', R('breiman1996', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006', 'hofstra2000',
      'jang1993', 'kingma2015', 'muntean2011', 'neumann2000', 'platt1999', 'porwal2004', 'raines2001', 'roberts2017',
      'stevens2004', 'takagi1985', 'yousefi2015')],
  ],
};
