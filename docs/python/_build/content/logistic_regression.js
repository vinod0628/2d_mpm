const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('logistic_regression_carlin');
const K = C.load('logistic_regression_curnamona');

const cfgOf = (X) => Object.assign({ design: 'all', estimator: 'firth', selection: 'none', neg_ratio: 5 }, X.selected);
const cj = cfgOf(J), ck = cfgOf(K);
const coefRows = (X) => X.reference.tables.coefficients.map((r) => [r.term, r.label === 'x' ? '' : (r.label || ''), g(r.B), r.s_B == null ? '—' : g(r.s_B, 3),
  r.z == null ? '—' : f(r.z, 2), r.p_value == null ? '—' : g(r.p_value, 2), r.odds_ratio == null ? '—' : g(r.odds_ratio, 3),
  r.B_std == null ? '—' : f(r.B_std, 3), r.note || '']);
const varRows = (X) => X.reference.tables.variables.map((r) => [r.layer, r.type, r.status, r.df == null ? '—' : String(r.df),
  r.lr_chi2 == null ? '—' : f(r.lr_chi2, 2), r.lr_p == null ? '—' : g(r.lr_p, 2), r.wald_p == null ? '—' : g(r.wald_p, 2),
  r.gvif_adj == null ? '—' : f(r.gvif_adj, 2)]);
const fitRows = (X, keep) => X.reference.tables.fit.filter((r) => keep.some((k) => r.item.startsWith(k))).map((r) => [r.item, r.value]);
const FIT_KEEP = ['Training design', 'Estimator', 'Training deposits', 'Study area', 'Observation rows', 'Coefficients',
  'Iterations', '−2 ln L', 'Model χ²', 'AIC', 'BIC', 'McFadden', 'Hosmer', 'Observed deposits', 'Predicted deposits',
  'Prior probability', 'Intercept correction'];
const hlTable = (X, caption) => ['table', {
  caption,
  head: ['Group', 'Cells', 'Observed deposits', 'Expected deposits', 'P(cell) range'],
  widths: [900, 1500, 1800, 1800, 3026],
  rows: X.reference.tables.hl.map((r) => [String(r.group), int(r.cells), f(r.observed, 0), f(r.expected, 2), `${g(r.p_min, 3)} – ${g(r.p_max, 3)}`]),
}];
const calib = (X) => `${X.reference.tables.stats.n_deposits} observed vs ${f(X.reference.tables.stats.predicted, 1)} predicted deposits`;
const designName = { all: 'all study-area cells (weighted LR)', sampled: 'deposits + random non-deposit cells' };
const estName = { firth: 'Firth bias-reduced', ml: 'maximum likelihood', ridge: 'ridge' };

function signs(X, meaning) {
  return X.reference.tables.coefficients.filter((r) => r.term !== '(intercept)' && r.s_B != null).map((r) => {
    const sig = r.p_value != null && r.p_value < 0.05;
    const dir = r.B > 0 ? 'raises' : 'lowers';
    const m = meaning[r.layer] ? meaning[r.layer](r.B > 0, sig) : '';
    return `**${r.term}** ${dir} the log-odds (B${r.B_std != null ? ' per s.d.' : ''} = ${f(r.B_std != null ? r.B_std : r.B, 3)}, p = ${g(r.p_value, 2)}, ${sig ? 'significant' : 'not significant'})${m ? ' — ' + m : ''}.`;
  });
}
const MEAN_C = {
  geology_host: (pos) => (pos ? 'the favourable silty-carbonate host is the chemical trap of the Carlin system; because every training deposit lies in it (quasi-complete separation) maximum likelihood would let this coefficient grow without bound, and the finite Firth estimate is a lower bound of a very strong effect (the LR test, not the Wald test, measures its significance)' : 'unexpected sign for the host rock'),
  dist_faults: (pos) => (pos ? 'once host rock and Sb are in the model, distance to faults adds no independent information: the Sb halo is itself centred on the fluid conduits, so the fault control is already expressed by Sb (single-layer AUC of the distance 0.35, i.e. deposits are near faults)' : 'proximity to faults (fluid pathways) raises prospectivity'),
  as_log_idw: (pos) => (pos ? 'As is part of the Carlin pathfinder halo' : 'with Sb in the model As acts as a suppressor variable: at a given Sb level, high As marks the broader, distal part of the halo and As-rich carbonaceous shales rather than the ore zone; on its own As is positively associated with the deposits (single-layer AUC 0.70)'),
  sb_log_idw: (pos) => (pos ? 'Sb is the strongest stream-sediment expression of the Carlin hydrothermal footprint (late stibnite and Sb-bearing sulfides of the inner halo)' : 'unexpected sign for Sb'),
};
const MEAN_K = {
  rad_th: (pos) => (pos ? 'Th maps the U–Th–REE-rich A-type granites of the Mount Painter area' : 'given U, Th is negative: the two channels map the same granites (collinearity, Table 6.17), and the model keeps the more selective U'),
  rad_u: (pos) => (pos ? 'U marks the hydrothermally overprinted granite and breccias that host the REE minerals; it shares the granite source with Th (collinearity, Table 6.17)' : 'given Th, U is negative (collinearity, Table 6.17)'),
  rad_k: (pos) => (pos ? 'K reflects K-feldspar-rich granite and K-metasomatism' : 'given Th and U, K is negative: K-rich sedimentary and metasedimentary units dilute the granite signal'),
  geochem_nb_log_idw: (pos) => (pos ? 'Nb is co-enriched with the REE in A-type systems' : 'given the radiometric channels, Nb adds no positive information (sparse surface samples)'),
  aster_silica_index: (pos) => (pos ? 'unexpected sign; with seven occurrences the silica index is not well constrained' : 'quartz-rich cover hides the basement, so low silica index (exposed granite and alteration) is favourable'),
};

const auc = (X, lab) => { const r = X.search.find((s) => s.label === lab); return r ? f(r.mean_test_auc) : '—'; };
const designReason = [
  'All study-area cells (weighted LR) use every barren cell and give probabilities per km² directly (predicted ≈ observed deposits).',
  cj.design === 'all' ? `Carlin: the weighted design was as good as the sampled ones on held-out deposits (Table 6.5) and is calibrated.` : `Carlin: ${designName[cj.design]} gave the best held-out AUC.`,
  ck.design === 'sampled' ? `Curnamona: deposits + 1:${f(K.selected.neg_ratio || 1, 0)} random non-deposit cells with ${'5'} repeats ranked the held-out occurrences best (LODO ${auc(K, K.selected.label)} vs ${auc(K, 'all cells, Firth (default)')} for all cells): with seven occurrences, averaging models over several barren samples smooths the coefficients, while the weighted design is dominated by the ≈ 17 000 barren cells; the price is a less calibrated probability (§6.8).` : 'Curnamona: the weighted design was best.',
].join(' ');
const estReason = [
  'Firth removes the first-order small-sample bias of maximum likelihood and keeps coefficients finite when a lithology or a value range holds no deposit (separation) — the usual situation with few, clustered deposits; its shrinkage towards zero is the conservative choice for exploration targeting.',
  'Carlin: all training deposits lie in the favourable host rock (quasi-complete separation), so maximum likelihood does not converge for the host-rock coefficient (B → ∞, Wald p ≈ 1); Firth gives finite, interpretable coefficients with the same held-out AUC (Table 6.5).',
  ck.estimator === 'firth' ? 'Curnamona: with seven occurrences (≈ 1.4 events per coefficient) maximum likelihood is strongly biased, and Firth is required.' : `Curnamona: ${estName[ck.estimator]} gave the best LODO AUC, but with seven occurrences Firth is recommended for stable coefficients.`,
].join(' ');

module.exports = {
  file: 'logistic_regression.docx',
  code: 'LR',
  short: 'Logistic regression',
  title: 'Logistic Regression (LR) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven statistical method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Logistic regression models the probability that a unit cell contains a deposit as a logistic function of a linear combination of evidence values (Agterberg, 1974; Agterberg et al., 1993; Bonham-Carter, 1994). Unlike weights of evidence, all coefficients are estimated jointly, so correlated evidence layers are not double-counted and no conditional-independence assumption is needed. MPM Studio offers two training designs — **weighted LR on all study-area cells** (the classic design of Agterberg et al., 1993, and ArcSDM) and **deposits + random non-deposit cells** (Carranza and Hale, 2001) with the rare-events prior correction (King and Zeng, 2001) — three estimators (maximum likelihood, Firth bias reduction, ridge), backward variable selection and a full set of diagnostics (menu *Data-driven ▸ Statistical ▸ Logistic Regression*).'],
    ['table', {
      caption: 'Table 1.1. Source files.',
      head: ['File', 'Role'],
      widths: [4200, 4826],
      rows: [
        ['`python/logic/data_driven/logistic_regression/model.py`', 'Estimation (IRLS, Firth, ridge), designs, selection, inference, diagnostics, prediction'],
        ['`python/logic/data_driven/logistic_regression/plugin.py`', 'Registration; run in the worker thread; outputs and tables'],
        ['`python/logic/data_driven/logistic_regression/dialogs.py`', 'Parameter and results dialogs'],
        ['`python/logic/common/training.py`', 'Train / test split, thinning, non-deposit sampling, training / test points'],
        ['`python/tests/test_logistic.py`', 'Verification against statsmodels, ArcSDM `sdmlr.exe`, direct optimisation, WofE'],
      ],
    }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 The logistic model'],
    ['p', 'For a binary response — a unit cell does ($y=1$) or does not ($y=0$) contain a deposit — logistic regression is the generalised linear model with the logit link: the log-odds of the deposit probability are a linear function of the evidence. A coefficient is the change of the log-odds per unit of its evidence variable with all other variables held constant; $e^{β}$ is the corresponding odds ratio. Continuous evidence enters as values (often log-transformed geochemistry, distances), categorical evidence as 0/1 dummy variables with a reference class.'],
    ['h2', '2.2 Logistic regression in mineral prospectivity mapping'],
    ['p', 'Agterberg (1974) introduced logistic regression for mapping mineral potential. Agterberg et al. (1993) showed that weights of evidence and logistic regression are two forms of the log-linear model and that **weighted** logistic regression on the unique conditions of the study area — each condition weighted by its area — removes the bias that conditional dependence causes in weights of evidence; for one binary pattern the LR coefficient equals the WofE contrast $C=W^{+}-W^{-}$ (verified in `test_logistic.py`). Carranza and Hale (2001) applied LR to deposit and non-deposit locations to map gold potential. Because deposits are rare, sampling all deposits and a fraction of the barren cells changes only the intercept (case-control sampling); King and Zeng (2001) give the prior correction that restores probabilities.'],
    ['h2', '2.3 Few deposits: separation and bias'],
    ['p', 'With few deposits, maximum-likelihood estimates are biased away from zero and, when a class or threshold contains no deposits (separation), do not exist — the coefficient tends to −∞. Firth (1993) penalises the likelihood with Jeffreys’ prior, which removes the first-order bias and always gives finite estimates; Heinze and Schemper (2002) established it as the remedy for separation. The rule of about ten events per estimated coefficient (Peduzzi et al., 1996) indicates when estimates are unstable.'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Model and likelihood'],
    ['eq', '\\rm{logit} P(\\bf{x})=\\ln\\frac{P}{1-P}=β_{0}+\\sum_{j=1}^{k}{β_{j}x_{j}}', 'lr-model'],
    ['p', 'With observation rows $i$ holding $a_{i}$ deposit cells out of $n_{i}$ cells (a unique condition, a single cell or a sample), $\\bf{X}$ the design matrix (intercept, z-scored continuous columns, dummies) and $p_{i}=\\rm{expit}(\\bf{x}_{i}^{T}β)$, the log-likelihood, score and Fisher information are'],
    ['eq', '\\ell(β)=\\sum_{i}{[a_{i}\\ln p_{i}+(n_{i}-a_{i})\\ln(1-p_{i})]}', 'lr-ll'],
    ['eq', 'U(β)=\\bf{X}^{T}(\\bf{a}-\\bf{n}p),      I(β)=\\bf{X}^{T}\\bf{W}\\bf{X},  \\bf{W}=\\rm{diag}(n_{i}p_{i}(1-p_{i}))', 'lr-score'],
    ['p', 'and the Newton–Raphson (IRLS) step is $β←β+I^{-1}U$, with step halving whenever the (penalised) log-likelihood decreases. Aggregating identical rows (unique conditions) leaves [[lr-ll]] unchanged, which is why the weighted LR of Agterberg et al. (1993) and the per-cell LR are the same model.'],
    ['h2', '3.2 Firth and ridge'],
    ['p', 'Firth’s estimator maximises the penalised likelihood $\\ell^{*}=\\ell+\\frac{1}{2}\\ln|I(β)|$ by solving the modified score equations'],
    ['eq', 'U^{*}(β)=\\bf{X}^{T}(\\bf{a}-\\bf{n}p+\\bf{h}⊙(\\frac{1}{2}-p))=0', 'lr-firth'],
    ['p', 'where $h_{i}$ are the diagonal elements of the hat matrix $\\bf{H}=\\bf{W}^{1/2}\\bf{X}I^{-1}\\bf{X}^{T}\\bf{W}^{1/2}$. The penalised likelihood-ratio test of a layer refits with its coefficients fixed at 0 but keeps the penalty of the full model (Heinze and Schemper, 2002). Ridge regression maximises $\\ell-\\frac{λ}{2}\\sum_{j≥1}{β_{j}^{2}}$ (intercept not penalised).'],
    ['h2', '3.3 Sampled design and prior correction'],
    ['p', 'If all $n_{d}$ deposit cells and $n_{0}$ random barren cells are used, only the intercept is biased. With $τ$ the deposit share of the study area and $\\bar{y}=n_{d}/(n_{d}+n_{0})$ the deposit share of the sample, the corrected intercept is (King and Zeng, 2001)'],
    ['eq', '\\tilde{β}_{0}=β_{0}-\\ln[\\frac{1-τ}{τ}·\\frac{\\bar{y}}{1-\\bar{y}}]', 'lr-prior'],
    ['p', 'With $R$ independent samples, coefficients and probabilities are pooled with Rubin’s (1987) rules: $\\bar{β}=\\frac{1}{R}\\sum{β_{r}}$ and total variance $T=\\bar{W}+(1+\\frac{1}{R})B$ (within + between samples).'],
    ['h2', '3.4 Probability per unit cell and its uncertainty'],
    ['p', 'Observations are raster cells of area $A_{c}$; the probability per unit cell of area $A_{u}$ and its delta-method standard deviation are'],
    ['eq', 'P_{u}=1-(1-P_{c})^{A_{u}/A_{c}},      s(P_{c})=P_{c}(1-P_{c})\\sqrt{\\bf{x}^{T}\\rm{Cov}(β)\\bf{x}}', 'lr-unit'],
    ['p', 'and the t-value is $t=P/s(P)$ (Agterberg et al., 1993; ArcSDM *LRTValue*); $t>1.645$ indicates a probability significantly above zero at the 95 % level.'],
    ['h2', '3.5 Inference and fit'],
    ['bullets', [
      'Wald test: $z_{j}=β_{j}/s(β_{j})$ with $s=\\sqrt{\\rm{diag}(I^{-1})}$; odds ratio $e^{β_{j}}$ with 95 % CI $e^{β_{j}±1.96s_{j}}$.',
      'Likelihood-ratio test of a layer (all its dummies): $χ^{2}=2(\\ell^{*}_{\\rm{full}}-\\ell^{*}_{\\rm{reduced}})$, df = number of its coefficients.',
      'Model χ² against the intercept-only model; AIC $=-2\\ell+2k$; BIC $=-2\\ell+k\\ln n$; McFadden pseudo-R² $=1-\\ell/\\ell_{0}$ (McFadden, 1974).',
      'Hosmer–Lemeshow test: cells sorted by $p$ into 10 groups of equal size, $C=\\sum_{g}{(O_{g}-E_{g})^{2}/[E_{g}(1-E_{g}/n_{g})]}$, χ² with 8 df (Hosmer et al., 2013).',
      'Calibration: observed vs predicted number of deposits $\\sum_{i}{P_{c,i}}$; with maximum likelihood and an intercept they are equal by construction.',
      'Collinearity: generalised variance-inflation factor $\\rm{GVIF}=\\det R_{11}\\det R_{22}/\\det R$ per layer, reported as $\\rm{GVIF}^{1/(2df)}$; > 2.2 (VIF > 5) flags strongly correlated layers (Fox and Monette, 1992).',
    ]],

    ['h1', '4 Algorithm'],
    ['steps', [
      '**Evidence and study area** — resample to one grid; exclude cells with missing evidence or impute the area-weighted mean (continuous) / class area fractions (dummies), as ArcSDM.',
      '**Design matrix** — intercept, continuous columns as z-scores (mean and s.d. of the study area; coefficients are back-transformed to original units), one dummy per non-reference class (reference = largest class); optional log10 transform.',
      '**Training rows** — *all cells*: one row per cell (or per unique condition when all evidence is categorical); above 1 000 000 cells, all deposit cells + a uniform random share $s$ of the others with intercept correction $\\ln s$. *Sampled*: training deposits + $r·n_{d}$ random non-deposit cells ≥ $d_{\\rm{min}}$ from every known deposit and ≥ spacing apart; prior correction [[lr-prior]]; $R$ repeats pooled.',
      '**Estimation** — IRLS / Newton–Raphson from $β_{0}=\\rm{logit}(\\bar{y})$, step ≤ 5, step halving, convergence when the largest change < 10⁻⁸; Firth [[lr-firth]] or ridge if chosen; columns without variation in the training rows are fixed at 0.',
      '**Variable selection (optional)** — backward elimination of whole layers by AIC or by the (penalised) LR test p-value > α.',
      '**Inference and diagnostics** — Wald and LR tests, odds ratios, GVIF, model fit, Hosmer–Lemeshow, K-S check of unique conditions; sampled design: confusion matrix, training AUC and cross-validation.',
      '**Prediction** — $P_{c}$, $s(P)$ and $t$ for every study cell in chunks; conversion [[lr-unit]]; outputs `post.tif`, `post_std.tif`, `tvalue.tif`, training / test points and tables.',
    ]],
    ['table', {
      caption: 'Table 4.1. Library and own code; verification.',
      head: ['Component', 'Implementation and verification'],
      widths: [3200, 5826],
      rows: [
        ['IRLS maximum likelihood', 'Own code (NumPy); equals statsmodels GLM Binomial to 10⁻⁷ (coefficients) and 10⁻⁶ (s.e.)'],
        ['Comparison with ArcSDM', 'On the case data of ArcSDM’s Fortran `sdmlr.exe` (deposits as successes, area as failures) the engine reproduces its coefficients within 5·10⁻⁶ and its probabilities, standard deviations and t-values to all printed decimals (ArcSDM, 2025)'],
        ['Firth estimator', 'Own code; equals a direct maximisation of $\\ell+\\frac{1}{2}\\ln|I|$ and the modified score is 0 at the estimate'],
        ['Weights of evidence link', 'One binary pattern: $β_{1}=C$ and $P$ = WofE posterior (test)'],
        ['Statistics', 'SciPy (χ², normal, K-S distributions)'],
      ],
    }],

    ['h1', '5 How to use'],
    ['p', '*Data-driven ▸ Statistical ▸ Logistic Regression…*. The training-data section (deposits, mask, hold-out 20 % spatially stratified, seed, thinning, absence points) is the same as for the other data-driven methods.'],
    ['table', {
      caption: 'Table 5.1. Evidence layers.',
      head: ['Column', 'Meaning and guidance'],
      widths: [2200, 6826],
      rows: [
        ['Use', 'Tick the evidence layers (≈ 10 deposits per coefficient; a categorical layer costs classes − 1 coefficients).'],
        ['Variable type', '*Continuous* (value; ordered classes such as distance classes can also be used as values) or *Categorical (dummy variables)* (one 0/1 column per class, largest class = reference).'],
        ['Transform', '*none* or *log10(x)* for positive, right-skewed data (raw geochemistry).'],
      ],
    }],
    ['table', {
      caption: 'Table 5.2. Training samples and model options.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2400, 1800, 4826],
      rows: [
        ['Training cells', 'all study-area cells', 'Weighted LR on every cell (probability per unit cell directly) or deposits + random non-deposit cells (Carranza and Hale, 2001).'],
        ['Non-deposit samples per deposit', '1', 'Sampled design: ratio $r$.'],
        ['Min. distance from deposits', '4 cells', 'Sampled design: non-deposit cells keep this distance from every known deposit.'],
        ['Min. spacing between samples', 'auto', 'Sampled design: even spread of the non-deposit cells.'],
        ['Random samples (repeats)', '10', 'Sampled design: $R$ draws pooled with Rubin’s rules.'],
        ['Cross-validation', 'leave-one-out', 'Sampled design: LOO (≤ 300 samples, else 10-fold), 3-fold, 10-fold or none.'],
        ['Prior correction', 'on', 'Sampled design: [[lr-prior]]; off = relative favourability.'],
        ['Estimator', 'Firth', 'Firth (few deposits, separation), maximum likelihood, ridge (λ).'],
        ['Variable selection', 'none', 'Backward elimination by AIC or by the LR test (remove if p > α, default 0.10).'],
        ['Missing evidence', 'exclude', 'Exclude cells or area-weighted mean (ArcSDM rule).'],
        ['Unit cell area', '1 km²', 'Area to which the probability refers.'],
        ['Outputs / prefix', 'probability + std + t-value / lr', 'Optional linear predictor (logit).'],
      ],
    }],
    ['p', 'The results dialog shows the training deposits, McFadden R², predicted / observed deposits, ROC AUC of training and test deposits and (sampled design) the cross-validated AUC, plus the tabs *Summary* (model fit), *Coefficients* (B, s(B), z, p, CI, odds ratio, B per s.d.), *Evidence layers* (Wald and LR tests, GVIF), *Goodness of fit* (Hosmer–Lemeshow groups), *Sample validation*, *Selection*, *Repeats* and *Correlation*.'],

    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),

    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, 'Table 6.4. Evidence of the Carlin logistic regression.'),
    C.searchTable(J, 'Table 6.5. Carlin — candidate settings (spatial 70 / 30 splits).'),
    ['p', `The selected setting is **${cj.label}** (${designName[cj.design]}, ${estName[cj.estimator]} estimator). Over 10 independent splits it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (range ${f(J.robust_summary.test_auc_min)}–${f(J.robust_summary.test_auc_max)}) with a training AUC of ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)}: the small gap shows that a model with five coefficients does not over-fit 23–24 training deposits.`],
    C.robustTable(J, 'Table 6.6. Carlin — robustness of the selected setting over 10 splits.'),
    C.curvesTable(J, 'Table 6.7. Carlin — reference run (seed 42).'),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — LR probability per km² with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) Wald z of the coefficients.'],
    ['table', {
      caption: 'Table 6.8. Carlin — coefficients of the reference model (continuous: per unit of the evidence value; B per s.d. = per standard deviation; categorical: relative to the reference class).',
      head: ['Term', 'Class', 'B', 's(B)', 'z', 'p', 'Odds ratio', 'B per s.d.', 'Note'],
      widths: [1500, 900, 1000, 900, 700, 900, 1000, 900, 1226],
      rows: coefRows(J),
    }],
    ['table', {
      caption: 'Table 6.9. Carlin — tests of the evidence layers.',
      head: ['Layer', 'Type', 'Status', 'df', 'LR χ²', 'LR p', 'Wald p', 'GVIF^(1/2df)'],
      widths: [1800, 1100, 1000, 600, 1100, 1100, 1100, 1226],
      rows: varRows(J),
    }],
    ['table', {
      caption: 'Table 6.10. Carlin — model fit (reference run).',
      head: ['Statistic', 'Value'], widths: [4526, 4500], rows: fitRows(J, FIT_KEEP),
    }],
    hlTable(J, 'Table 6.11. Carlin — Hosmer–Lemeshow groups (deciles of the fitted cell probability).'),
    ['p', 'Interpretation of the coefficients: ' + signs(J, MEAN_C).join(' ')],

    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, 'Table 6.12. Evidence of the Curnamona logistic regression.'),
    C.searchTable(K, 'Table 6.13. Curnamona — candidate settings (leave-one-deposit-out).'),
    C.lodoTable(K, 'Table 6.14. Curnamona — leave-one-deposit-out results of the selected setting.'),
    C.curvesTable(K, 'Table 6.15. Curnamona — reference run with all seven occurrences.'),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — LR probability per km² with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) Wald z.'],
    ['table', {
      caption: 'Table 6.16. Curnamona — coefficients of the reference model.',
      head: ['Term', 'Class', 'B', 's(B)', 'z', 'p', 'Odds ratio', 'B per s.d.', 'Note'],
      widths: [1700, 700, 1000, 900, 700, 900, 1000, 900, 1226],
      rows: coefRows(K),
    }],
    ['table', {
      caption: 'Table 6.17. Curnamona — tests of the evidence layers and collinearity.',
      head: ['Layer', 'Type', 'Status', 'df', 'LR χ²', 'LR p', 'Wald p', 'GVIF^(1/2df)'],
      widths: [2000, 1100, 1000, 600, 1000, 1000, 1000, 1326],
      rows: varRows(K),
    }],
    ['table', {
      caption: 'Table 6.18. Curnamona — model fit (reference run).',
      head: ['Statistic', 'Value'], widths: [4526, 4500], rows: fitRows(K, FIT_KEEP),
    }],
    hlTable(K, 'Table 6.19. Curnamona — Hosmer–Lemeshow groups.'),
    ['p', 'Interpretation of the coefficients: ' + signs(K, MEAN_K).join(' ') + ' With seven occurrences and five coefficients (events per variable ≈ 1.4) the individual coefficients are uncertain and correlated radiometric channels share the signal; the probability map is nevertheless well ranked (LODO AUC), which is what prospectivity mapping needs.'],

    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: 'Table 6.20. Ideal logistic-regression parameters for the demo projects and their meaning.',
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1700, 1400, 1400, 4526],
      rows: [
        ['Evidence', 'host rock (dummy), distance to faults, log As, log Sb', 'radiometric Th, U, K; log Nb; silica index', 'One proxy per mineral-system component (pathway, trap, footprint; source granite, cover). The logit is linear in **log** concentration, matching the log-normal distribution of trace elements and the multiplicative build-up of a hydrothermal anomaly.'],
        ['Training cells', designName[cj.design], designName[ck.design], designReason],
        ['Estimator', estName[cj.estimator], estName[ck.estimator], estReason],
        ['Variable selection', cj.selection === 'none' ? 'none' : 'backward LR test', ck.selection === 'none' ? 'none' : 'backward LR test', 'Every layer represents a component of the ore-forming system; removing a weak but geologically required component (e.g. faults) makes the model fit the sample rather than the process. Selection was not better on held-out deposits.'],
        ['Variable types', 'host = categorical; others continuous', 'all continuous', 'Lithology is nominal; distances and concentrations keep their full resolution so that the logistic curve, not a chosen class boundary, expresses the gradual decrease of prospectivity away from faults and anomaly centres.'],
        ['Missing evidence', 'exclude', 'exclude', 'Cells outside the mapped host-rock units (Carlin) have no geological information; excluding them avoids inventing a lithology.'],
        ['Unit cell', '1 km²', '1 km²', 'Of the order of a deposit footprint (orebody plus inner alteration halo); probabilities per km² are directly comparable with the WofE posterior.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** Mean test AUC on Carlin ${f(J.robust_summary.test_auc_mean)} (10 splits); mean LODO AUC on Curnamona ${f(K.search.find((s) => s.label === K.selected.label).mean_test_auc)}.`,
      '**Joint estimation.** Correlated layers (As and Sb; Th and U) share the explained variation; the coefficients of correlated layers must be read together, and GVIF identifies them.',
      `**Calibration.** Predicted vs observed deposits: Carlin ${calib(J)} (${designName[cj.design]}), Curnamona ${calib(K)} (${designName[ck.design]}). With the weighted design maximum likelihood reproduces the observed number exactly and Firth’s penalty adds a few deposits; the sampled design restores the intercept by the prior correction only, and with seven occurrences its probabilities are inflated — use the weighted design when absolute probabilities matter, the sampled one for ranking. Hosmer–Lemeshow p-values: Carlin ${f(J.reference.tables.stats.hl_p, 2)}, Curnamona ${f(K.reference.tables.stats.hl_p, 2)} (p > 0.05: no evidence of lack of fit).`,
      '**Linearity.** The model is linear in the logit; thresholds and interactions must be supplied as evidence (classified layers, products). Random forest captures them automatically.',
    ]],

    ['h1', '7 References'],
    ['refs', R('agterberg1974', 'agterberg1993', 'bonham1994', 'carranza2001', 'carranza2008', 'chung2003', 'cline2005',
      'delong1988', 'deepseam', 'elburg2012', 'fawcett2006', 'firth1993', 'fox1992', 'heinze2002', 'hofstra2000',
      'hosmer2013', 'king2001', 'mcfadden1974', 'muntean2011', 'neumann2000', 'peduzzi1996', 'raines2001', 'roberts2017',
      'rubin1987', 'stevens2004', 'yousefi2015', 'zuo2020', 'arcsdm')],
  ],
};
