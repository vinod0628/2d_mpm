const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('bayesian_network_carlin');
const K = C.load('bayesian_network_curnamona');
const WJ = C.has('wofe_carlin') ? C.load('wofe_carlin') : null;
const P = (X) => X.reference.tables.params;
const st = (X) => X.reference.tables.stats;
const tab = (X, k) => X.reference.tables[k];
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const cand = (X, lab) => { const r = X.search.find((s) => s.label.startsWith(lab)); return r ? r.mean_test_auc : null; };
let tn = 3;
const T = () => ++tn;
const edgeRows = (X) => tab(X, 'edges').map((r) => [r.parent, r.child, f(r.cond_mutual_information, 4)]);
const layRows = (X) => tab(X, 'layers').map((r) => [r.layer, String(r.classes), f(r.mutual_information_with_deposits, 4),
  r.parent, f(r.auc_drop_without_layer, 3)]);
const cmpRows = (X) => tab(X, 'comparison').map((r) => [r.structure, f(r.log_likelihood, 1), String(r.parameters), f(r.bic, 1),
  f(r.training_auc), String(r.edges)]);
const cptRows = (X, layer) => tab(X, 'cpt').filter((r) => r.layer === layer && !r.parent_class).map((r) => [r.cls,
  f(r.p_given_deposit, 4), f(r.p_given_barren, 4), f(r.log_ratio, 3), String(r.deposits), int(r.cells)]);
const ratio = (X) => (st(X).predicted ? X.dataset_info && st(X).deposits / st(X).predicted : null);
function edgeText(X, ds) {
  const e = tab(X, 'edges');
  if (!e.length) return 'Naive Bayes was selected: the evidence layers are treated as conditionally independent.';
  const top = [...e].sort((a, b) => b.cond_mutual_information - a.cond_mutual_information)[0];
  const why = ds === 'carlin'
    ? (/(as_log|sb_log)/.test(top.parent) && /(as_log|sb_log)/.test(top.child) ? 'As and Sb are deposited from the same Carlin fluids and dispersed together into the stream sediments — weights of evidence counts this one process twice.' : 'the two layers share a geological control.')
    : (/rad_/.test(top.parent) && /rad_/.test(top.child) ? 'both radiometric channels map the same U–Th–REE-rich granites.' : 'the two layers share a geological control.');
  return `The strongest learned dependency is **${top.parent} → ${top.child}** (I = ${f(top.cond_mutual_information, 3)} nats): ${why}`;
}

module.exports = {
  file: 'bayesian_network.docx',
  code: 'BN',
  short: 'Bayesian network',
  title: 'Bayesian Network Classifiers for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven Bayesian method of MPM Studio — naive Bayes and tree-augmented naive Bayes',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'A Bayesian network represents the joint probability distribution of the deposit variable and the evidence layers by a directed acyclic graph and conditional probability tables. Porwal, Carranza and Hale (2006) introduced Bayesian network classifiers to mineral potential mapping and showed that the **tree-augmented naive Bayes** (TAN) classifier, which models dependencies between evidence layers, outperforms naive Bayes — which, trained on all cells, is exactly the weights-of-evidence model. MPM Studio (*Data-driven ▸ Statistical ▸ Bayesian Network*) learns both structures, the TAN dependency tree by the conditional mutual information (Friedman et al., 1997), estimates the tables with a Dirichlet prior and propagates their uncertainty into the posterior map. The code is own NumPy / networkx code (pgmpy, the usual Python library, requires PyTorch and was therefore not used).'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4300, 4726], rows: [
      ['`python/logic/data_driven/bayesian_network/model.py`', 'Discretisation, TAN structure, Dirichlet tables, posterior, uncertainty, comparison'],
      ['`python/logic/data_driven/bayesian_network/plugin.py`, `dialogs.py`', 'Registration, outputs, dialogs with the network drawing'],
      ['`python/tests/test_bayesian_network.py`', 'Naive Bayes = WofE, conditional MI, TAN dependency and calibration, Carlin model'],
    ] }],
    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Bayesian networks'],
    ['p', 'A Bayesian network factorises the joint distribution as the product of the probabilities of every node given its parents in the graph. For classification only the posterior of the class node is needed; restricting the graph to the class node as parent of every evidence node gives naive Bayes, which assumes the evidence layers independent given the class — the conditional-independence assumption of weights of evidence (Bonham-Carter, 1994).'],
    ['h2', '2.2 Tree-augmented naive Bayes'],
    ['p', 'Friedman, Geiger and Goldszmidt (1997) allow every evidence node one additional parent among the evidence nodes. The optimal tree (maximum likelihood) is the maximum-weight spanning tree of the conditional mutual information I(Xᵢ; Xⱼ | D) (Chow and Liu, 1968). Pairs of layers that carry the same information about the deposits — e.g. two pathfinder elements of one fluid — are linked, and their joint evidence is counted once.'],
    ['h2', '2.3 Bayesian estimation and uncertainty'],
    ['p', 'Tables are estimated from counts with a Dirichlet prior (equivalent sample size α spread evenly over the table cells, BDeu-type; Heckerman et al., 1995), which avoids zero probabilities for classes without deposits. The posterior distribution of every table is again Dirichlet; drawing tables from it and recomputing the deposit posterior gives its uncertainty.'],
    ['h1', '3 Mathematical basis'],
    ['eq', 'P(D=1|\\bf{x})∝P(D=1)\\prod_{i}{P(x_{i}|D=1,x_{π(i)})}', 'bn-post'],
    ['p', 'with $π(i)$ the evidence parent of layer $i$ (none for naive Bayes and for the TAN root).'],
    ['eq', 'I(X_{i};X_{j}|D)=\\sum_{d}{\\sum_{a,b}{P(a,b,d)\\ln\\frac{P(a,b|d)}{P(a|d)P(b|d)}}}', 'bn-cmi'],
    ['eq', 'θ_{ijk}=\\frac{N_{ijk}+α/(r_{i}q_{i})}{N_{ij}+α/q_{i}}', 'bn-cpt'],
    ['p', '($r_{i}$ classes of layer $i$, $q_{i}$ configurations of its parents within a class of D). Naive Bayes with α → 0 on all cells: $\\ln O(D|\\bf{x})=\\ln O(D)+\\sum_{i}{\\ln\\frac{P(x_{i}|D)}{P(x_{i}|\\bar{D})}}$ — the weights-of-evidence posterior with the class weights $W^{+}$ (verified in the tests). Uncertainty: $θ^{(s)}∼\\rm{Dir}(N+α/(rq))$, $s=1…S$, posterior s.d. over the draws. Continuous layers: $k$ equal-area (quantile) classes.'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Discretisation** — continuous layers → $k$ quantile classes; categorical layers keep their classes; missing → own class or excluded.',
      '**Training cells** — all study cells (deposit cells = 1) or deposits + random non-deposit cells.',
      '**Structure** — naive Bayes, or TAN: conditional MI of all pairs [[bn-cmi]], maximum spanning tree (networkx), directed from the first layer.',
      '**Tables** — Dirichlet-smoothed counts [[bn-cpt]].',
      '**Posterior** — [[bn-post]] for every cell; per unit cell $P_{u}=1-(1-P)^{A_{u}/A_{c}}$; Dirichlet draws for the s.d. (evaluated once per unique evidence combination).',
      '**Diagnostics** — NB vs TAN log-likelihood, parameters, BIC, AUC; mutual information of every layer; AUC drop without each layer; observed vs predicted deposits.',
    ]],
    ['h1', '5 How to use'],
    ['table', { caption: 'Table 5.1. Settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2400, 2200, 4426], rows: [
      ['Structure', 'TAN', 'TAN or naive Bayes.'],
      ['Training cells', 'all study-area cells', 'All cells (calibrated probabilities, as WofE) or deposits + random non-deposit cells (ratio, min. distance).'],
      ['Classes of continuous layers', '5', 'Equal-area classes; fewer with few deposits.'],
      ['Dirichlet prior α', '1', 'Equivalent sample size of the tables.'],
      ['Uncertainty draws', '50', 'Dirichlet draws for the posterior s.d. (0 = off).'],
      ['Unit cell / missing evidence', '1 km² / own class', ''],
      ['Outputs / prefix', 'posterior + per unit cell + s.d. / bn', ''],
    ] }],
    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),
    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, `Table 6.${T()}. Evidence of the Carlin Bayesian network.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (spatial 70 / 30 splits).`),
    ['p', `The selected setting is **${J.selected.label}**. Over 10 splits it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (training ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)})${WJ ? ` — weights of evidence with classified evidence: ${f(WJ.robust_summary.test_auc_mean)}` : ''}.`],
    C.robustTable(J, `Table 6.${T()}. Carlin — robustness over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42).`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — Bayesian network posterior probability with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) drop of the training AUC without each layer.'],
    ['table', { caption: `Table 6.${T()}. Carlin — TAN dependencies (parent → child).`, head: ['Parent', 'Child', 'I(parent; child | D)'], widths: [3000, 3000, 3026], rows: edgeRows(J) }],
    ['p', edgeText(J, 'carlin')],
    ['table', { caption: `Table 6.${T()}. Carlin — evidence layers.`, head: ['Layer', 'Classes', 'I(X; D)', 'Parents', 'AUC drop without'], widths: [2200, 1000, 1500, 2526, 1800], rows: layRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — naive Bayes vs TAN on the same cells.`, head: ['Structure', 'Log-likelihood', 'Parameters', 'BIC', 'Training AUC', 'Edges'], widths: [1700, 1600, 1300, 1600, 1500, 1326], rows: cmpRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — probability table of the host rock (root of the tree).`, head: ['Class', 'P(class | deposit)', 'P(class | barren)', 'ln ratio', 'Deposits', 'Cells'], widths: [1800, 1600, 1600, 1200, 1100, 1726], rows: cptRows(J, 'geology_host') }],
    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, `Table 6.${T()}. Evidence of the Curnamona Bayesian network.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — Bayesian network posterior with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) AUC drop without each layer.'],
    ['table', { caption: `Table 6.${T()}. Curnamona — TAN dependencies.`, head: ['Parent', 'Child', 'I(parent; child | D)'], widths: [3000, 3000, 3026], rows: edgeRows(K) }],
    ['p', edgeText(K, 'curnamona') + ` Mean LODO AUC of the selected setting: **${f(lodo(K))}**.`],
    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal Bayesian-network parameters for the demo projects.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1600, 1400, 1400, 4626],
      rows: [
        ['Structure', P(J).structure.toUpperCase(), P(K).structure.toUpperCase(), 'TAN models the dependency of proxies of one process (As–Sb of one fluid; Th–U of one granite) instead of double-counting it; the selection compares both on held-out deposits (naive Bayes: ' + f(cand(J, 'naive Bayes')) + ' Carlin, ' + f(cand(K, 'naive Bayes')) + ' Curnamona).'],
        ['Classes of continuous layers', String(P(J).n_classes), String(P(K).n_classes), 'Every class of a TAN child is conditioned on the class of its parent and on D; with few deposits few classes keep the tables populated. Geologically, 3–5 classes resolve background, weak and strong anomaly or near / intermediate / far from faults.'],
        ['Training cells', P(J).design === 'all' ? 'all study cells' : 'sampled', P(K).design === 'all' ? 'all study cells' : 'sampled', 'All cells give calibrated probabilities per cell like WofE and use every barren cell; the sampled design only re-weights the barren class.'],
        ['Dirichlet α', g(P(J).alpha, 2), g(P(K).alpha, 2), 'A weak prior (α = 1) protects classes without deposits from probability 0 — a class without known deposits is unexplored rather than impossible.'],
        ['Missing evidence', 'own class', 'own class', 'Unmapped ground keeps its area; its posterior is driven by the other layers, and the class “missing” carries the information that mapped vs unmapped ground may differ.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** Mean test AUC on Carlin ${f(J.robust_summary.test_auc_mean)}; mean LODO AUC on Curnamona ${f(lodo(K))}.`,
      '**Conditional independence.** TAN repairs the main weakness of weights of evidence — the double counting of correlated evidence — and its posterior probabilities are better calibrated (predicted ≈ observed deposits in the GUI test, 0.95 vs 0.59 for WofE on Carlin).',
      '**Interpretability.** The network drawing and the probability tables show which layers are linked and how each class shifts the odds — as transparent as weights of evidence.',
      '**Limitations.** Discretisation loses detail within classes; one parent per layer cannot represent higher-order dependencies.',
    ]],
    ['h1', '7 References'],
    ['refs', R('bonham1994', 'chow1968', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006',
      'friedman1997', 'heckerman1995', 'hofstra2000', 'muntean2011', 'neumann2000', 'porwal2006', 'raines2001', 'roberts2017',
      'stevens2004', 'yousefi2015')],
  ],
};
