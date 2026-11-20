const { R } = require('../refs');
const C = require('../common');

const { f, g } = C;
const J = C.load('fuzzy_logic_carlin');
const K = C.load('fuzzy_logic_curnamona');
const T_ = (X, k) => X.reference.tables[k];
const auc = (X) => X.reference.curves.training.auc;
let tn = 3;
const T = () => ++tn;

const memRows = (X) => T_(X, 'memberships').map((r) => [r.layer, r.group, r.membership, String(r.parameters).replace(/[{}()]/g, ''),
  f(r.mean, 3), f(100 * r.area_above_0_5, 1)]);
const netRows = (X) => T_(X, 'inference_network').map((r) => [r.node, r.inputs, r.operator, r.gamma == null ? '—' : g(r.gamma, 2), f(r.mean, 3)]);
const cand = (X, lab) => { const r = X.search.find((s) => s.label.startsWith(lab)); return r ? f(r.mean_test_auc) : '—'; };

module.exports = {
  file: 'fuzzy_logic.docx',
  code: 'FUZ',
  short: 'Fuzzy logic overlay',
  title: 'Fuzzy Logic Overlay for Mineral Prospectivity Mapping',
  subtitle: 'Knowledge-driven method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Fuzzy logic overlay is the most widely used knowledge-driven method of mineral prospectivity mapping (An et al., 1991; Bonham-Carter, 1994; Porwal et al., 2003b). The geologist translates the deposit model into fuzzy memberships — the degree, between 0 and 1, to which each value of an evidence layer is favourable — and combines them with fuzzy operators in an **inference network** that mirrors the mineral system: proxies of the same process are combined first (e.g. alternative pathfinder elements of one fluid), then the processes (source, pathway, trap, footprint) are combined into favourability (Wyborn et al., 1994; McCuaig et al., 2010). Deposits are not needed to build the map and serve only to validate it, so the method works in green-field areas without known deposits. MPM Studio: *Knowledge-driven ▸ Fuzzy ▸ Fuzzy Logic Overlay*.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4300, 4726], rows: [
      ['`python/logic/knowledge_driven/fuzzy_logic/model.py`', 'Membership functions (scikit-fuzzy), operators, inference network'],
      ['`python/logic/knowledge_driven/fuzzy_logic/plugin.py`, `dialogs.py`', 'Registration, outputs, dialogs with data-based default parameters and membership plots'],
      ['`python/tests/test_knowledge_overlays.py`', 'Membership functions, operators (= eis_toolkit), inference network, Carlin model'],
    ] }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Fuzzy sets'],
    ['p', 'In classical (crisp) sets an element belongs to a set or not; in a fuzzy set (Zadeh, 1965) it belongs to a degree μ ∈ [0, 1]. “Favourable for Carlin-type gold” is such a set: a cell 300 m from a fault is more favourable than one 3 km away, but there is no sharp boundary. Fuzzy memberships express the expert’s graded judgement and avoid the arbitrary thresholds of Boolean logic.'],
    ['h2', '2.2 Membership functions'],
    ['p', 'Ordered evidence uses monotone functions — increasing for pathfinder concentrations or radiometric intensities, decreasing for distances to structures — or bell-shaped functions where an intermediate value is optimal; nominal evidence (lithology) gets a membership per class. The logistic function makes the transition smooth and needs only an inflection point and a slope (Yousefi and Carranza, 2015a).'],
    ['h2', '2.3 Fuzzy operators'],
    ['p', 'The fuzzy AND (minimum) is conservative — favourability is limited by the weakest evidence; the fuzzy OR (maximum) is liberal — any one strong evidence suffices. The algebraic product decreases and the algebraic sum increases with every added layer. The γ operator (Zimmermann and Zysno, 1980) is a compromise between sum and product controlled by γ ∈ [0, 1]; the geometric average does not depend on the number of layers (Yousefi and Carranza, 2015b). In MPM, γ between 0.7 and 0.95 is the usual choice for combining independent processes (Bonham-Carter, 1994).'],
    ['h2', '2.4 Inference networks and mineral systems'],
    ['p', 'The mineral-systems approach decomposes ore formation into necessary components — a metal and fluid source, an energy driver, pathways and a trap, and a preserved footprint (Wyborn et al., 1994; McCuaig et al., 2010). An inference network combines the proxies of each component with an operator appropriate to their relation (OR for alternative proxies of the same component, AND / γ for proxies that must coincide) and then combines the components — all must be present, so AND or high γ.'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Membership functions'],
    ['eq', 'μ_{lin}=\\rm{clip}(\\frac{x-a}{b-a},0,1),      μ_{log}=\\frac{1}{1+e^{-s(x-i)}},      μ_{gauss}=\\exp(-\\frac{(x-c)^{2}}{2σ^{2}})', 'fz-mf'],
    ['eq', 'μ_{S}=2(\\frac{x-a}{b-a})^{2}  (a<x≤\\frac{a+b}{2}),      μ_{S}=1-2(\\frac{x-b}{b-a})^{2}  (\\frac{a+b}{2}<x<b),      μ_{S}=0 (x≤a), 1 (x≥b)', 'fz-s'],
    ['p', 'μ_Z = 1 − μ_S; ArcGIS “large” μ = 1/(1 + (x/m)^(−f)) and “small” μ = 1/(1 + (x/m)^f); per class μ(c) as entered. Default parameters are taken from the 2nd and 98th percentiles $p_{2}$, $p_{98}$ of the study area (logistic: $i=(p_{2}+p_{98})/2$, $s=9.2/(p_{98}-p_{2})$, mapping that range to 0.01–0.99).'],
    ['h2', '3.2 Operators'],
    ['eq', 'μ_{AND}=\\rm{min}_{k}μ_{k},  μ_{OR}=\\rm{max}_{k}μ_{k},  μ_{prod}=\\prod_{k}{μ_{k}},  μ_{sum}=1-\\prod_{k}{(1-μ_{k})}', 'fz-ops'],
    ['eq', 'μ_{γ}=μ_{sum}^{γ}·μ_{prod}^{1-γ},      μ_{geo}=(\\prod_{k}{μ_{k}})^{1/n}', 'fz-gamma'],
    ['h2', '3.3 Inference network'],
    ['eq', 'F(\\bf{x})=⊗_{final}(⊗_{g_{1}}(μ_{k}(x_{k}):k∈g_{1}),…,⊗_{g_{m}}(μ_{k}(x_{k}):k∈g_{m}))', 'fz-net'],
    ['p', 'where $⊗$ is the operator of each group and of the final combination.'],

    ['h1', '4 Algorithm'],
    ['steps', [
      '**Study area** — mask; cells with missing evidence excluded (or membership 0.5 = no information).',
      '**Memberships** — every layer transformed by its function [[fz-mf]] / [[fz-s]] (scikit-fuzzy `smf`, `zmf`, `sigmf`, `gaussmf`; own code for the others).',
      '**Groups** — layers with the same group name combined by the group operator [[fz-ops]] / [[fz-gamma]].',
      '**Final combination** — the groups combined by the final operator [[fz-net]].',
      '**Outputs and validation** — favourability, group and membership maps; ROC, success-rate and P–A curves of all deposits.',
    ]],
    ['table', { caption: 'Table 4.1. Verification.', head: ['Component', 'Verification'], widths: [3000, 6026], rows: [
      ['Membership functions', 'Analytic values (inflection = 0.5, S + Z = 1, Gaussian at ±σ = e^(−½), large / small symmetry) (test)'],
      ['Operators', 'γ = 0 equals product, γ = 1 equals sum; γ and sum equal eis_toolkit’s `gamma_overlay` / `sum_overlay` (test)'],
      ['Inference network', 'Favourability = min(max(μ_a, μ_c), μ_b) for an OR group inside an AND network (test)'],
    ] }],

    ['h1', '5 How to use'],
    ['p', '*Knowledge-driven ▸ Fuzzy ▸ Fuzzy Logic Overlay…*.'],
    ['table', { caption: 'Table 5.1. Settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2400, 2000, 4626], rows: [
      ['Study area mask / deposits / absence points', 'mask; deposits (validation only)', 'Deposits are optional: without them the map is produced but not validated.'],
      ['Membership function', 'linear (continuous) / per class (categorical)', 'Linear, S, Z, logistic, Gaussian, large, small or per class.'],
      ['Parameters', 'from the 2–98 % range of the data', 'Edit them to express the deposit model (e.g. distance at which fault control vanishes, anomaly threshold).'],
      ['Group', '1', 'Layers with the same group name form one mineral-system component.'],
      ['Within each group', 'fuzzy AND', 'AND / gamma for proxies that must coincide; OR for alternative proxies of one process.'],
      ['Between groups / γ', 'gamma / 0.9', 'All components must be present: gamma 0.8–0.95 or AND.'],
      ['Missing evidence', 'exclude', 'Exclude cells or give membership 0.5.'],
      ['Outputs / prefix', 'favourability + group maps / fuzzy', 'Optional membership maps.'],
    ] }],
    ['p', 'Results: inference-network table, membership-function plots over the data histograms, membership statistics, ROC / AUC of all deposits.'],

    ['h1', '6 Demo data: deposit models, results and their geological meaning'],
    ...C.demoData(6),
    ...C.protocolKD(6),

    ['h2', '6.5 Carlin (Au)'],
    C.searchTable(J, `Table 6.${T()}. Carlin — inference-network configurations (all 32 deposit cells).`),
    ['table', { caption: `Table 6.${T()}. Carlin — memberships of the selected model.`, head: ['Layer', 'Group', 'Function', 'Parameters', 'Mean μ', 'Area μ ≥ 0.5 (%)'], widths: [1700, 1200, 1100, 2526, 1200, 1300], rows: memRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — inference network.`, head: ['Node', 'Inputs', 'Operator', 'γ', 'Mean'], widths: [1700, 3226, 1500, 1000, 1600], rows: netRows(J) }],
    C.curvesTable(J, `Table 6.${T()}. Carlin — validation of the selected model with all deposits.`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — fuzzy favourability with the known deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success rate, (c) P–A plot, (d) mean membership of the evidence layers.'],
    ['p', `Without any training, the deposit model reaches a ROC AUC of **${f(auc(J))}**. The configurations differ little (Table 6.4: ${cand(J, 'OR within groups, gamma 0.7')} for γ = 0.7 to ${cand(J, 'OR within groups, geometric')} for the geometric average): the information is in the memberships — which values of each proxy the deposit model calls favourable — more than in the operator.`],

    ['h2', '6.6 Curnamona (REE)'],
    C.searchTable(K, `Table 6.${T()}. Curnamona — inference-network configurations (all 7 occurrences).`),
    ['table', { caption: `Table 6.${T()}. Curnamona — memberships of the selected model.`, head: ['Layer', 'Group', 'Function', 'Parameters', 'Mean μ', 'Area μ ≥ 0.5 (%)'], widths: [1900, 1200, 1100, 2326, 1200, 1300], rows: memRows(K) }],
    ['table', { caption: `Table 6.${T()}. Curnamona — inference network.`, head: ['Node', 'Inputs', 'Operator', 'γ', 'Mean'], widths: [1700, 3226, 1500, 1000, 1600], rows: netRows(K) }],
    C.curvesTable(K, `Table 6.${T()}. Curnamona — validation with all occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — fuzzy favourability with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) mean membership.'],
    ['p', `The REE deposit model reaches a ROC AUC of **${f(auc(K))}** with all seven occurrences — comparable with the data-driven methods (LODO AUC 0.87–0.90 in the companion documents) although nothing was trained on the occurrences.`],

    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Deposit models of the demo projects and the reasoning behind every membership and operator.`,
      head: ['Element', 'Carlin (Au)', 'Curnamona (REE)', 'Geological and ore-forming reasoning'],
      widths: [1500, 1800, 1800, 3926],
      rows: [
        ['Components (groups)', 'trap, pathway, footprint', 'source, enrichment, exposure', 'Mineral-systems approach (McCuaig et al., 2010): every component is necessary; each group holds the proxies of one process.'],
        ['Trap / source', 'host rock: favourable 0.9, other 0.1', 'Th, U: logistic, increasing', 'Carlin: reactive silty carbonates neutralise and sulfidise the fluid — gold precipitates there and almost nowhere else (Cline et al., 2005). Curnamona: U–Th–REE-rich A-type granite is the metal source (Neumann et al., 2000); higher radiometric counts = more source rock at surface.'],
        ['Pathway', 'distance to faults: Z-shaped, 500 → 5000 m', '—', 'High-angle faults channelled the ore fluid; within ≈ 0.5 km of a mapped fault favourability is full, beyond ≈ 5 km the fault control has vanished (lateral fluid flow along permeable beds reaches a few km).'],
        ['Footprint / enrichment', 'As, Sb: logistic, increasing; OR within the group', 'Nb: logistic, increasing', 'As and Sb record the same hydrothermal halo — either suffices, so they are combined with OR (taking both with AND would double-count one process). Nb is co-enriched with the REE in A-type systems.'],
        ['Exposure', '—', 'silica index: logistic, decreasing', 'Quartz-rich sand / silcrete cover hides the basement; low silica index = exposed granite and alteration.'],
        ['Between components', 'γ = ' + (J.selected.gamma || 0.9) + ' (' + (J.selected.final_op || 'gamma') + ')', 'γ = ' + (K.selected.gamma || 0.9) + ' (' + (K.selected.final_op || 'gamma') + ')', 'High γ: all components are required (AND-like), but strong evidence of one component partly compensates weak evidence of another — appropriate because mapping of each component is incomplete (unmapped faults, covered host rock).'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy without training.** ROC AUC of all deposits: Carlin ${f(auc(J))}, Curnamona ${f(auc(K))} — expert knowledge alone ranks the known deposits well.`,
      '**Transparency.** Every number on the map can be traced to a membership and an operator: the map is a formalised deposit model and can be defended to explorers and reviewers.',
      '**Choice of operator.** The differences between reasonable operators are small; the memberships (thresholds of the deposit model) matter most.',
      '**Subjectivity.** Memberships are expert judgements; their sensitivity should be tested (here by the configuration tables). Data-driven calibration of memberships is a hybrid extension (fuzzy weights of evidence; logistic fuzzification).',
    ]],

    ['h1', '7 References'],
    ['refs', R('an1991', 'bonham1994', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006',
      'hofstra2000', 'mccuaig2010', 'muntean2011', 'neumann2000', 'porwal2003b', 'raines2001', 'wyborn1994', 'yousefi2015',
      'yousefi2015a', 'yousefi2015b', 'zadeh1965', 'zimmermann1980')],
  ],
};
