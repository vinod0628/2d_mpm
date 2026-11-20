const { R } = require('../refs');
const C = require('../common');

const { f } = C;
const J = C.load('boolean_logic_carlin');
const K = C.load('boolean_logic_curnamona');
const T_ = (X, k) => X.reference.tables[k];
const auc = (X) => X.reference.curves.training.auc;
const comb = (X) => T_(X, 'conditions').find((r) => r.layer === 'FAVOURABLE (combined)') || {};
let tn = 3;
const T = () => ++tn;
const condRows = (X) => T_(X, 'conditions').map((r) => [r.layer, r.group || '', r.condition, String(r.values).replace(/[()]/g, ''),
  f(r.area_pct, 1), String(r.deposits_inside), f(r.deposits_pct, 1), r.ratio == null ? '—' : f(r.ratio, 2)]);
const netRows = (X) => T_(X, 'network').map((r) => [r.node, r.inputs, r.operator, f(r.area_pct, 1)]);

module.exports = {
  file: 'boolean_logic.docx',
  code: 'BOOL',
  short: 'Boolean logic overlay',
  title: 'Boolean Logic Overlay for Mineral Prospectivity Mapping',
  subtitle: 'Knowledge-driven method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Boolean logic overlay is the simplest and oldest GIS method of mineral prospectivity mapping (Bonham-Carter, 1994): every evidence layer is turned into a binary map — favourable or not — by a condition taken from the deposit model, and the binary maps are combined with Boolean AND / OR. It reproduces the exploration geologist’s checklist (“within 3 km of a fault, in the favourable host, and anomalous in Sb”) and outlines target areas directly. MPM Studio (*Knowledge-driven ▸ Overlay ▸ Boolean Logic Overlay*) organises the conditions in mineral-system groups (AND / OR within and between groups) and also outputs the number of satisfied conditions, which ranks the cells and can be validated by ROC.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4300, 4726], rows: [
      ['`python/logic/knowledge_driven/boolean_logic/model.py`', 'Conditions, AND / OR network, statistics'],
      ['`python/logic/knowledge_driven/boolean_logic/plugin.py`, `dialogs.py`', 'Registration, outputs, dialogs'],
      ['`python/tests/test_knowledge_overlays.py`', 'Conditions, groups, area / deposit statistics, Carlin model'],
    ] }],
    ['h1', '2 Fundamental theory'],
    ['p', 'Boolean logic assumes that every criterion of the deposit model is either fulfilled or not, with equal importance. AND (intersection) requires all criteria and yields small, conservative target areas — it fails wherever one layer is incompletely mapped; OR (union) requires any criterion and yields large areas. Grouping conditions by mineral-system component combines both: OR among alternative proxies of one component, AND among necessary components (Wyborn et al., 1994; McCuaig et al., 2010). The method cannot rank cells within the favourable area — the count of satisfied conditions provides a coarse ranking (an index overlay with equal weights, Bonham-Carter, 1994).'],
    ['p', 'The quality of a Boolean target map is judged by the share of deposits it contains relative to its share of the study area: a ratio of 1 is random, larger ratios mean the conditions concentrate the deposits.'],
    ['h1', '3 Mathematical basis'],
    ['eq', 'B_{k}(\\bf{x})=[x_{k}≥t_{k}] or [x_{k}≤t_{k}] or [a_{k}≤x_{k}≤b_{k}] or [x_{k}∈S_{k}]', 'bool-cond'],
    ['eq', 'G_{g}=∧_{k∈g}B_{k}  or  ∨_{k∈g}B_{k},      F=∧_{g}G_{g}  or  ∨_{g}G_{g},      N(\\bf{x})=\\sum_{k}{B_{k}(\\bf{x})}', 'bool-net'],
    ['eq', 'ρ=\\frac{n_{D}(F)/n_{D}}{A(F)/A},      \\rm{AUC}(N)=P(N(\\bf{x}_{D})>N(\\bf{x}_{\\bar{D}}))+\\frac{1}{2}P(=)', 'bool-ratio'],
    ['p', 'with $n_{D}(F)$ the deposits inside the favourable area $F$ of area $A(F)$; ties of the integer count $N$ count ½ in the AUC.'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Study area** — mask ∩ cells with all evidence (a condition on missing evidence is undefined).',
      '**Conditions** — every layer → binary map [[bool-cond]].',
      '**Groups and combination** — [[bool-net]]; number of satisfied conditions $N$.',
      '**Statistics and validation** — area, deposits and ratio [[bool-ratio]] of every condition and of the favourable area; ROC / success rate / P–A of $N$.',
    ]],
    ['h1', '5 How to use'],
    ['table', { caption: 'Table 5.1. Settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2400, 2000, 4626], rows: [
      ['Condition', '≥ (continuous) / in classes (categorical)', '≥, ≤, between, outside, in classes.'],
      ['Value(s)', 'quartiles of the data', 'Thresholds of the deposit model in evidence units.'],
      ['Group', '1', 'Mineral-system component of the condition.'],
      ['Within each group / between groups', 'AND / AND', 'OR within groups of alternative proxies; AND between necessary components.'],
      ['Outputs / prefix', 'favourable area + conditions met / boolean', 'Optional condition maps.'],
    ] }],
    ['h1', '6 Demo data: deposit models, results and their geological meaning'],
    ...C.demoData(6),
    ...C.protocolKD(6),
    ['p', 'For the Boolean overlay the configurations are ranked by the share of deposits captured minus the share of area used by the favourable map (Youden’s J = TPR − FPR of the binary map), because the condition count can reward a degenerate checklist whose target area is tiny.'],
    ['h2', '6.5 Carlin (Au)'],
    C.searchTable(J, `Table 6.${T()}. Carlin — configurations (all 32 deposit cells; ROC of the condition count).`),
    ['table', { caption: `Table 6.${T()}. Carlin — conditions of the selected model.`, head: ['Layer', 'Group', 'Condition', 'Value(s)', 'Area %', 'Deposits', 'Deposits %', 'Ratio'], widths: [1700, 1100, 900, 1426, 900, 900, 1100, 1000], rows: condRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — combination.`, head: ['Node', 'Inputs', 'Operator', 'Area %'], widths: [2000, 4026, 1500, 1500], rows: netRows(J) }],
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — Boolean favourable area with the known deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success rate, (c) P–A plot of the condition count, (d) deposit / area ratio of the conditions.'],
    ['p', `The selected checklist outlines ${f(comb(J).area_pct, 1)} % of the study area containing ${f(comb(J).deposits_pct, 0)} % of the deposits (ratio ${f(comb(J).ratio, 2)}); the condition count reaches a ROC AUC of ${f(auc(J))}. The favourable host alone already holds ${f(T_(J, 'conditions')[0].deposits_pct, 0)} % of the deposits in ${f(T_(J, 'conditions')[0].area_pct, 0)} % of the area — the chemical trap is the most selective single criterion of the Carlin system.`],
    ['h2', '6.6 Curnamona (REE)'],
    C.searchTable(K, `Table 6.${T()}. Curnamona — configurations (all 7 occurrences).`),
    ['table', { caption: `Table 6.${T()}. Curnamona — conditions of the selected model.`, head: ['Layer', 'Group', 'Condition', 'Value(s)', 'Area %', 'Deposits', 'Deposits %', 'Ratio'], widths: [1900, 1100, 900, 1226, 900, 900, 1100, 1000], rows: condRows(K) }],
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — Boolean favourable area with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) deposit / area ratio of the conditions.'],
    ['p', `The selected checklist outlines ${f(comb(K).area_pct, 1)} % of the area with ${f(comb(K).deposits_pct, 0)} % of the occurrences (ratio ${f(comb(K).ratio, 2)}); condition-count AUC ${f(auc(K))}.`],
    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Boolean deposit models of the demo projects.`,
      head: ['Element', 'Carlin (Au)', 'Curnamona (REE)', 'Geological and ore-forming reasoning'],
      widths: [1500, 1800, 1800, 3926],
      rows: [
        ['Trap / source', 'host rock = favourable class', 'Th, U ≥ threshold (OR)', 'Carlin: gold precipitated where the fluid met reactive silty carbonate — a necessary condition. Curnamona: radioactive granite is the REE source; Th and U are alternative proxies of the same rock (OR).'],
        ['Pathway', 'distance to faults ≤ ' + (J.selected.faults_m || 3000) + ' m', '—', 'Fluid conduits: deposits lie within a few km of mapped faults (lateral flow along permeable beds).'],
        ['Footprint / enrichment', 'As or Sb ≥ upper ' + (100 - (J.selected.q || 75)) + ' %', 'Nb ≥ upper ' + (100 - (K.selected.q || 75)) + ' %', 'The As–Sb halo is the hydrothermal footprint (OR: either element suffices); Nb marks HFSE–REE co-enrichment.'],
        ['Exposure', '—', 'silica index ≤ threshold', 'Low-silica ground = granite and alteration not hidden by quartz-rich cover.'],
        ['Combination', 'OR within, AND between', 'OR within, AND between', 'Alternative proxies of one process are interchangeable; all processes must have operated for a deposit to form.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Target areas.** Carlin: ${f(comb(J).area_pct, 1)} % of the area with ${f(comb(J).deposits_pct, 0)} % of the deposits; Curnamona: ${f(comb(K).area_pct, 1)} % with ${f(comb(K).deposits_pct, 0)} %.`,
      '**Strictness.** Upper-decile thresholds give small, rich targets but miss deposits whose footprint is weak or unmapped; median thresholds capture most deposits in large areas — the configuration tables quantify this trade-off.',
      '**Limitations.** Equal weights, sharp thresholds and no ranking inside the target area; fuzzy logic or data-driven methods refine the Boolean map.',
    ]],
    ['h1', '7 References'],
    ['refs', R('bonham1994', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006', 'hofstra2000',
      'mccuaig2010', 'muntean2011', 'neumann2000', 'raines2001', 'wyborn1994', 'yousefi2015')],
  ],
};
