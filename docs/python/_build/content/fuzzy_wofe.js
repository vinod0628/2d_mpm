const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('fuzzy_wofe_carlin');
const K = C.load('fuzzy_wofe_curnamona');
const tab = (X) => X.reference.tables;
const st = (X) => tab(X).stats;
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const cand = (X, start) => { const r = X.search.find((s) => s.label.startsWith(start)); return r ? r.mean_test_auc : null; };
const zeroB = (X) => tab(X).layers.filter((r) => r.beta <= 1e-6).map((r) => r.layer);
const modelOf = (X) => (X.selected.model === 'classic' ? 'classic' : 'modified');
let tn = 3;
const T = () => ++tn;

const knowTable = (X, cap) => ['table', {
  caption: cap, head: ['Layer', 'Process', 'Direction / class scores', 'Expert weight'], widths: [2600, 1800, 3326, 1300],
  rows: X.knowledge.map((r) => [r.layer, r.process, r.direction === '+' ? 'high values favourable' : r.direction === '−' ? 'low values favourable' : r.direction, g(r.weight, 2)]),
}];
const layerTable = (X, cap) => ['table', {
  caption: cap,
  head: ['Layer', 'Pattern B: classes / area % / deposits', 'W⁺', 'W⁻', 'C', 'Stud(C)', 'β ± s.e. (modified)'],
  widths: [1700, 1900, 900, 900, 900, 900, 1826],
  rows: tab(X).layers.map((r) => [r.layer, `${r.pattern_classes} of ${r.classes} / ${f(r.pattern_area_pct, 1)} / ${r.pattern_deposits}`,
    f(r.w_plus, 2), f(r.w_minus, 2), f(r.contrast, 2), f(r.studentized_contrast, 2),
    r.beta <= 1e-6 ? '0 (bound)' : pm(r.beta, r.beta_se, 2)]),
}];
const weightTable = (X, cap, layer) => ['table', {
  caption: cap, head: ['Class', 'μ', 'Cells', 'Deposits', 'In B', 'W(μ) ± s.d.'], widths: [2626, 1000, 1400, 1100, 900, 2000],
  rows: tab(X).weights.filter((r) => r.layer === layer).map((r) => [r.cls, f(r.membership, 2), int(r.cells), String(r.deposits),
    r.in_pattern ? 'yes' : '', pm(r.fuzzy_weight, r.fuzzy_weight_sd, 2)]),
}];
const ciTable = (X, cap) => ['table', { caption: cap, head: ['Item', 'Value'], widths: [5026, 4000], rows: tab(X).ci_test.map((r) => [r.item, r.value]) }];

const interp = (X, name) => {
  const z = zeroB(X), s = st(X);
  const t = [`The selected setting is **${X.selected.label}**.`];
  t.push(`The classic model predicts ${f(s.S_classic, 1)} deposits for ${s.deposits} observed (N/S = ${f(s.ci_ratio, 2)}): ${s.ci_ratio > 1.15 ? 'its probabilities are too low because the expert memberships do not span 0–1 and therefore compress the contrast of every layer' : s.ci_ratio < 0.85 ? 'its probabilities are too high because correlated layers are counted twice' : 'close to the observed number'}; the modified model reproduces N exactly (Σ P = ${f(s.S_modified, 1)}).`);
  t.push(`Mean modified coefficient ${f(s.mean_beta, 2)}${s.mean_beta > 1.2 && s.ci_ratio > 1.15 ? ' (> 1: the logistic regression stretches the compressed fuzzy weights back to the contrast the deposits support)' : s.mean_beta > 1.2 ? ' (the coefficients redistribute the weight among the layers: strong layers are stretched, overlapping ones shrunk)' : ''}.`);
  if (modelOf(X) === 'classic') {
    const c = cand(X, 'classic'), m = cand(X, 'modified, expert memberships, max. contrast pattern, 10 classes (default)');
    t.push(`The classic posterior is selected (${f(c)} versus ${f(m)} for the modified model): with only ${s.deposits} deposits the ${tab(X).layers.length} extra coefficients of the modified model are poorly determined (large standard errors), so the fixed weights of the classic model rank the held-out occurrences better even though its probabilities are mis-scaled.`);
  }
  if (z.length) t.push(`${z.join(' and ')} ${z.length > 1 ? 'get' : 'gets'} β = 0: once the other layers are in the model ${z.length > 1 ? 'they add' : 'it adds'} no information in the expert direction (Table ${name}).`);
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
    ['Process weights', 'all 1 (not used by fuzzy WofE)', 'all 1 (not used by fuzzy WofE)'],
    ['Posterior model', modelOf(J), modelOf(K)],
    ['Class memberships', J.selected.membership || 'expert', K.selected.membership || 'expert'],
    ['Favourable pattern B', (J.selected.pattern || 'max_contrast') === 'half' ? 'μ ≥ 0.5' : 'maximum studentized contrast', (K.selected.pattern || 'max_contrast') === 'half' ? 'μ ≥ 0.5' : 'maximum studentized contrast'],
    ['Modified model: β ≥ 0', J.selected.nonnegative === false ? 'off' : 'on', K.selected.nonnegative === false ? 'off' : 'on'],
    ['Classes per continuous layer', String(J.selected.n_classes || 10), String(K.selected.n_classes || 10)],
    ['Output process maps / prefix', 'off / fwofe', 'off / fwofe'],
    ['Result', `test AUC ${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)} (10 splits)`, `LODO AUC ${f(lodo(K))}`],
  ] }];

module.exports = {
  file: 'fuzzy_wofe.docx',
  code: 'FWofE',
  short: 'Fuzzy weights of evidence',
  title: 'Fuzzy Weights of Evidence (Classic and Modified) for Mineral Prospectivity Mapping',
  subtitle: 'Hybrid method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Weights of evidence (WofE; Bonham-Carter et al., 1989) turns binary evidence patterns into additive log-odds weights counted from known deposits. Fuzzy weights of evidence (Cheng and Agterberg, 1999) generalises the binary pattern to a fuzzy set: every evidence class has a membership μ in the favourable pattern, and its weight is a mixture of the weights of the pattern and of its complement. When μ comes from expert knowledge and the conditional probabilities from the deposits, the method is a knowledge–data hybrid (hybrid fuzzy WofE; Porwal et al., 2006). The modified fuzzy WofE (Agterberg, 2011; Zhang et al., 2014; Zhang and Agterberg, 2018) keeps the fuzzy weights but estimates one coefficient per layer by logistic regression, which removes the bias of the conditional-independence assumption, reproduces the observed number of deposits and still handles missing data. MPM Studio: *Hybrid ▸ Fuzzy evidence ▸ Fuzzy Weights of Evidence*.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4200, 4826], rows: [
      ['`python/logic/hybrid/fuzzy_wofe/model.py`', 'Classes, memberships, favourable pattern, fuzzy weights and variances, classic and modified posterior, independence test'],
      ['`python/logic/hybrid/fuzzy_wofe/plugin.py`, `dialogs.py`', 'Registration, outputs, dialog'],
      ['`python/logic/hybrid/common/*`', 'Knowledge table, robust scaling and expert memberships shared by the hybrid methods'],
      ['`python/tests/test_fuzzy_wofe.py`', 'W⁺/W⁻ limits and variances, grouped fit = logistic regression, β ≥ 0, Σ P = N, dependence deflation, data memberships, missing data, Carlin model'],
    ] }],
    ['note', 'Sources of the implementation: the fuzzy weight [[fw-w]] and the class-wise conditional probabilities follow Cheng and Agterberg (1999) as implemented in the open GeoDAS 3D-WofE scripts (Intelligent Exploration, n.d.); the expert memberships follow the hybrid model of Porwal et al. (2006); the modified posterior [[fw-mod]] implements the principle stated by Agterberg (2011), Zhang et al. (2014) and Zhang and Agterberg (2018) — a WofE model whose layer coefficients are estimated by (weighted) logistic regression — as a logistic regression of the deposit indicator on the fuzzy weights. The β ≥ 0 option is an MPM Studio design choice for hybrid models (Section 2.5).'],
    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Weights of evidence'],
    ['p', 'For a binary pattern B and deposits D, the positive and negative weights $W^{+}=\\ln[P(B|D)/P(B|\\bar{D})]$ and $W^{-}=\\ln[P(\\bar{B}|D)/P(\\bar{B}|\\bar{D})]$ are the log-likelihood ratios of the pattern being present or absent; the contrast $C=W^{+}-W^{-}$ measures the association and $C/s(C)$ its significance. Under conditional independence of the layers given the deposits, Bayes\' rule gives the posterior log-odds as the prior log-odds plus one weight per layer (Bonham-Carter et al., 1989; Bonham-Carter, 1994).'],
    ['h2', '2.2 Fuzzy patterns'],
    ['p', 'A multi-class layer (distance to faults, a geochemical anomaly) is reduced in classical WofE to one binary pattern at an optimal threshold, which discards the gradation inside and outside the pattern. Cheng and Agterberg (1999) treat the favourable pattern as a fuzzy set: a cell belongs to it with membership μ ∈ [0, 1], and its weight interpolates — in probability, not in log-odds — between the fully favourable (μ = 1, weight W⁺) and the fully unfavourable state (μ = 0, weight W⁻). Fuzzy WofE therefore keeps the ordering of all classes while the conditional probabilities are still counted from one crisp pattern, which remains robust with few deposits.'],
    ['h2', '2.3 Knowledge-based memberships (hybrid fuzzy WofE)'],
    ['p', 'Porwal et al. (2006) assign the memberships from expert knowledge — logistic functions of the expert class scores — and the conditional probabilities from the deposits, so the model combines a knowledge-driven ordering of the evidence with data-driven weights. MPM Studio uses the knowledge network of its hybrid methods: continuous layers are scaled robustly in the expert direction and turned into memberships σ(1.5(u − 0.5)); categorical layers (e.g. a geology map) receive an expert score per class; the membership of a class is the mean membership of its cells. A data-driven alternative sets μ from the contrast of every class (logistic, minimum → 0.01, maximum → 0.99), as in the GeoDAS scripts.'],
    ['h2', '2.4 Conditional independence and the modified model'],
    ['p', 'Adding weights assumes that the layers are conditionally independent given the deposits. Correlated layers (two pathfinder elements of one halo, a host unit and its geophysical signature) are then counted twice and the probabilities are too high; Agterberg and Cheng (2002) test this by comparing the predicted number of deposits S = Σ P with the observed N. Logistic regression estimates the coefficients jointly and is unbiased whatever the dependence, but loses the interpretability of WofE and cannot handle missing data. The modified WofE (Agterberg, 2011; Zhang et al., 2014; Zhang and Agterberg, 2018) combines the two: the weights remain the evidence scores, a logistic regression estimates one coefficient per layer, S = N holds by construction and missing evidence keeps the weight 0. For fuzzy weights the coefficient β also corrects the scale: memberships that do not span 0–1 compress W(μ), and β > 1 stretches it back.'],
    ['h2', '2.5 Keeping the expert direction'],
    ['p', 'With collinear layers an unconstrained logistic regression can give a layer a negative coefficient — its weights are then used against the expert direction (e.g. "far from faults is favourable" once As and Sb are known). In a hybrid model this contradicts the knowledge the weights were built from. MPM Studio therefore constrains β ≥ 0 by default (bounded maximum likelihood): a layer that would get a negative coefficient gets β = 0 and is reported as adding no information beyond the other layers. The unconstrained fit remains available and is compared in Section 6.'],
    ['h1', '3 Mathematical basis'],
    ['p', 'For each layer, classes c with $n_{c}$ cells and $d_{c}$ deposit cells; the favourable pattern B is the set of classes with the highest memberships cut at the maximum studentized contrast (or at μ ≥ 0.5). With $a=n(B∩D)$, $b=n(\\bar{B}∩D)$, $c=n(B∩\\bar{D})$, $d=n(\\bar{B}∩\\bar{D})$ (an empty count is replaced by ½),'],
    ['eq', 'P(B|D)=\\frac{a}{a+b},  P(\\bar{B}|D)=\\frac{b}{a+b},  P(B|\\bar{D})=\\frac{c}{c+d},  P(\\bar{B}|\\bar{D})=\\frac{d}{c+d}', 'fw-p'],
    ['eq', 'W(μ)=\\ln\\frac{μ P(B|D)+(1-μ)P(\\bar{B}|D)}{μ P(B|\\bar{D})+(1-μ)P(\\bar{B}|\\bar{D})}', 'fw-w'],
    ['p', '$W(1)=W^{+}$ and $W(0)=W^{-}$. With Poisson counts — the assumption behind the classical variances (Bonham-Carter, 1994) — the delta method gives'],
    ['eq', '\\rm{Var}W(μ)=\\frac{μ^{2}a+(1-μ)^{2}b}{(μ a+(1-μ)b)^{2}}+\\frac{μ^{2}c+(1-μ)^{2}d}{(μ c+(1-μ)d)^{2}}', 'fw-var'],
    ['p', 'which reduces to $1/a+1/c$ for μ = 1 and $1/b+1/d$ for μ = 0. Memberships: expert $μ_{c}=\\rm{mean}_{i∈c}σ(1.5(u_{i}-0.5))$, $u=d(x-\\rm{med})/\\rm{IQR}$ (categorical: the expert score); data $μ_{c}=σ(s(C_{c}-i))$ with $i=(C_{min}+C_{max})/2$, $s=2\\ln 99/(C_{max}-C_{min})$ and $C_{c}$ the contrast of class c against the rest. Posterior per cell (missing evidence: $W_{j}=0$):'],
    ['eq', '\\rm{logit}P_{classic}=\\rm{logit}\\frac{N}{n}+\\sum_{j}{W_{j}(μ_{j,c(i)})}', 'fw-classic'],
    ['eq', '\\rm{logit}P_{mod}=β_{0}+\\sum_{j}{β_{j}W_{j}(μ_{j,c(i)})},      β_{j}≥0', 'fw-mod'],
    ['p', 'The β maximise the binomial likelihood over all cells; because $W_{j}$ takes one value per class, the cells are grouped by their class combination g with $n_{g}$ cells and $k_{g}$ deposit cells:'],
    ['eq', '\\ln L(β)=\\sum_{g}{[k_{g}η_{g}-n_{g}\\ln(1+e^{η_{g}})]},      η_{g}=β_{0}+\\sum_{j}{β_{j}W_{gj}}', 'fw-lik'],
    ['p', 'At the maximum $\\sum_{i}{P_{mod,i}}=N$ (score equation of β₀). Standard errors from the observed information (undefined for β on the bound). Conditional-independence test (Agterberg and Cheng, 2002): $T=S-N$ with $S=\\sum_{i}{P_{classic,i}}$ and $\\rm{Var}(S)≈\\sum_{i}{[P_{i}(1-P_{i})]^{2}\\rm{Var}(\\rm{logit}P_{i})}$, $\\rm{Var}(\\rm{logit}P_{i})=\\sum_{j}{\\rm{Var}W_{j}}$.'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Knowledge network** — robust directed scaling and expert memberships of all study cells (no deposits used).',
      '**Classes** — categorical classes, or quantile classes of a continuous layer (default 10).',
      '**Memberships** — expert (mean membership / class score) or data (logistic of the class contrast).',
      '**Favourable pattern** — classes sorted by membership; cumulative pattern with the maximum studentized contrast (at least one deposit), or μ ≥ 0.5.',
      '**Fuzzy weights** — [[fw-w]] and [[fw-var]] for every class.',
      '**Classic posterior** [[fw-classic]] and the independence test.',
      '**Modified posterior** [[fw-mod]] by bounded maximum likelihood [[fw-lik]] over the class combinations.',
      '**Outputs** — posterior of the selected model, the other model, s.d. of the log-odds, expert map, limiting process.',
    ]],
    ['h1', '5 How to use'],
    ['table', { caption: 'Table 5.1. Settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2300, 1900, 4826], rows: [
      ['Deposits / mask / hold-out / thinning', 'deposit layer, project mask, 20 %, off', 'As for the other data-driven methods; non-deposits are not sampled (every cell is counted).'],
      ['Knowledge table', '—', 'Process (only for the expert map and the limiting process), direction of continuous layers, expert score 0–1 per class of categorical layers (classes without a score get 0.05). The expert weights are not used by fuzzy WofE — the deposits set the weights.'],
      ['Posterior model', 'modified', 'Modified: β by logistic regression (Σ P = N, dependence deflated). Classic: Σ W (conditional independence assumed).'],
      ['Class memberships', 'expert', 'Expert = hybrid fuzzy WofE; data = memberships from the class contrasts (data-driven fuzzy WofE).'],
      ['Favourable pattern B', 'maximum studentized contrast', 'Crisp pattern of W⁺ / W⁻; μ ≥ 0.5 follows the expert threshold instead.'],
      ['Modified model: β ≥ 0', 'on', 'Keeps every layer in its expert direction; β = 0 marks a layer that adds nothing beyond the others.'],
      ['Classes per continuous layer', '10', 'Fewer with few deposits (each class needs deposits for a stable count of its pattern).'],
    ] }],
    ['table', { caption: 'Table 5.2. Outputs.', head: ['Output', 'Content'], widths: [2300, 6726], rows: [
      ['`post`', 'Posterior probability per cell of the selected model.'],
      ['`post_classic` / `post_modified`', 'The other model.'],
      ['`logit_sd`', 's.d. of the classic posterior log-odds from the counting errors of the weights [[fw-var]].'],
      ['`expert`, `limiting_process`', 'Expert favourability and weakest process of the knowledge network.'],
      ['Tables', '`layers` (pattern, W⁺, W⁻, C, Stud(C), β ± s.e.), `weights` (class, μ, cells, deposits, W(μ) ± s.d.), `ci_test` (N, S, N/S, Agterberg–Cheng test).'],
    ] }],
    ['p', 'Reading the results: β = 1 uses the fuzzy weights as they are; β > 1 means the memberships understate the evidence; β = 0 means the layer adds nothing in its expert direction once the others are known. N/S of the classic model below 0.85 points to dependent layers, above 1.15 to compressed memberships.'],
    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),
    ['h2', '6.5 Carlin (Au)'],
    knowTable(J, `Table 6.${T()}. Carlin — knowledge table.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (5 spatial 70 / 30 splits).`),
    C.robustTable(J, `Table 6.${T()}. Carlin — selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42).`),
    layerTable(J, `Table 6.${T()}. Carlin — favourable patterns, weights and modified coefficients (reference run).`),
    weightTable(J, `Table 6.${T()}. Carlin — fuzzy weights of sb_log_idw.`, 'sb_log_idw'),
    ciTable(J, `Table 6.${T()}. Carlin — conditional independence.`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — fuzzy WofE posterior probability with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) fuzzy weights against membership.'],
    ['p', `${interp(J, `6.${tn - 2}`)} Over 10 splits the selected setting reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}**. Classic versus modified: ${f(cand(J, 'classic'))} versus ${f(cand(J, 'modified, expert memberships, max. contrast pattern, 10 classes (default)'))}; unconstrained β: ${f(cand(J, 'modified, unconstrained'))}; data memberships: ${f(cand(J, 'modified, data memberships'))}.`],
    ['h2', '6.6 Curnamona (REE)'],
    knowTable(K, `Table 6.${T()}. Curnamona — knowledge table.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all occurrences.`),
    layerTable(K, `Table 6.${T()}. Curnamona — favourable patterns, weights and modified coefficients.`),
    ciTable(K, `Table 6.${T()}. Curnamona — conditional independence.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — fuzzy WofE posterior probability with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) fuzzy weights.'],
    ['p', `${interp(K, `6.${tn - 1}`)} Mean LODO AUC of the selected setting: **${f(lodo(K))}**. With seven occurrences every pattern rests on a handful of deposit cells, so the fuzzy weights carry large counting errors (output \`logit_sd\`, variance [[fw-var]]); few classes keep the patterns stable.`],
    ['h2', '6.7 Recommended setup for the demo projects'],
    ['p', 'The settings below reproduce the selected results from the fuzzy WofE dialog; Section 6.8 explains the reasoning.'],
    setupTable(`Table 6.${T()}. Fuzzy WofE dialog settings for the demo projects.`),
    ['h2', '6.8 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal fuzzy WofE settings.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1500, 1600, 1600, 4326],
      rows: [
        ['Evidence / processes', 'host (trap), faults (pathway), As + Sb (deposition)', 'Th + U (source), Nb (deposition), silica (footprint)', 'One to two proxies per mineral-system process: for Carlin-type gold the reactive host, the fault conduits and the As–Sb halo; for the REE system the U–Th–REE granite source, the co-enriched Nb and the quartz-rich cover that hides it.'],
        ['Posterior model', modelOf(J), modelOf(K), `Carlin (${st(J).deposits} training deposits): As and Sb belong to one hydrothermal halo and both follow the host rock and faults, so their information overlaps; the modified model estimates their joint effect and recalibrates the compressed weights (classic N/S ${f(st(J).ci_ratio, 2)}), with the same ranking as the classic model (${f(cand(J, 'classic'))} vs ${f(cand(J, 'modified, expert memberships, max. contrast pattern, 10 classes (default)'))}). Curnamona (${st(K).deposits} occurrences): radiometric Th and U come from the same granites (classic N/S ${f(st(K).ci_ratio, 2)}, probabilities too high), but seven occurrences cannot determine one coefficient per layer, so the classic weights rank better (LODO ${f(cand(K, 'classic'))} vs ${f(cand(K, 'modified, expert memberships, max. contrast pattern, 10 classes (default)'))}); use the modified model there for calibrated probabilities, the classic one for ranking.`],
        ['Memberships', J.selected.membership || 'expert', K.selected.membership || 'expert', 'Expert memberships encode the deposit model (closer to faults, richer in pathfinders, inside the host = more favourable); the deposits only set how strongly each layer counts.'],
        ['Favourable pattern', (J.selected.pattern || 'max_contrast') === 'half' ? 'μ ≥ 0.5' : 'max. contrast', (K.selected.pattern || 'max_contrast') === 'half' ? 'μ ≥ 0.5' : 'max. contrast', 'The pattern with the most significant contrast is where the mineralising process is actually expressed (e.g. the fault corridor that captured the fluids), not an arbitrary expert threshold.'],
        ['Classes', String(J.selected.n_classes || 10), String(K.selected.n_classes || 10), 'Enough classes to follow gradual footprints (dispersion halos, distance to conduits), few enough that each pattern holds several deposits.'],
        ['β ≥ 0', J.selected.nonnegative === false ? 'off' : 'on', K.selected.nonnegative === false ? 'off' : 'on', `A layer used against its geological meaning would be an artefact of collinearity; with the bound ${zeroB(J).length ? zeroB(J).join(', ') + ' get β = 0 on Carlin' : 'no Carlin layer hits the bound'}${zeroB(K).length ? ' and ' + zeroB(K).join(', ') + ' on Curnamona' : ''}.`],
      ],
    }],
    ['h2', '6.9 Discussion'],
    ['bullets', [
      `**Accuracy.** Carlin test AUC ${f(J.robust_summary.test_auc_mean)} (10 splits); Curnamona mean LODO AUC ${f(lodo(K))}.`,
      '**Fuzzy versus binary.** Fuzzy weights keep the gradation of every layer while the conditional probabilities still come from one robust crisp pattern per layer.',
      '**Classic versus modified.** The classic posterior is easy to explain but mis-scaled whenever the layers are dependent or the memberships compressed; the modified posterior is calibrated (Σ P = N) and shows through β which layers carry independent information.',
      '**Knowledge versus data.** The expert decides the order of the classes; the deposits decide the size of the weights and, through β, how much each layer counts.',
      '**Limits.** The ordering is only as good as the expert directions; a wrong direction gives a negative contrast, which the layer table and the warnings show.',
    ]],
    ['h1', '7 References'],
    ['refs', R('agterberg2011', 'agtcheng2002', 'bonham1989', 'bonham1994', 'cheng1999', 'chung2003', 'cline2005', 'deepseam',
      'delong1988', 'elburg2012', 'fawcett2006', 'geodas3d', 'hofstra2000', 'muntean2011', 'neumann2000', 'porwal2006f',
      'raines2001', 'roberts2017', 'stevens2004', 'yousefi2015', 'zhang2014', 'zhang2018')],
  ],
};
