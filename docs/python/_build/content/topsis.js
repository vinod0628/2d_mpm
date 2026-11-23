const { R } = require('../refs');
const C = require('../common');

const { f, g } = C;
const J = C.load('topsis_carlin');
const K = C.load('topsis_curnamona');
const ev = (X) => X.reference.tables.evidence;
const auc = (X) => X.reference.curves.training.auc;
const tr = (X) => X.reference.curves.training;
const cand = (X, start) => { const r = X.search.find((s) => s.label.startsWith(start)); return r ? r.mean_test_auc : null; };
const candW = (X, start) => { const r = X.search.find((s) => s.label.startsWith(start)); return r && r.weights ? r.weights : {}; };
const io = (X) => C.load(`index_overlay_${X.dataset}`);
const ioAuc = (X) => io(X).reference.curves.training.auc;
const ruleTxt = (r) => ({ classes: 'score per class', positive: 'linear, high values favourable', negative: 'linear, low values favourable', ge: 'binary ≥', le: 'binary ≤' }[r] || r);
const specTxt = (r) => (r.rule === 'positive' || r.rule === 'negative' ? r.scores.replace(' … ', ', ') : r.scores);
const evRows = (X) => ev(X).map((r) => [r.layer, ruleTxt(r.rule), r.scores, f(r.weight, 3), f(r.mean_score, 3),
  f(r.favourable_area_pct, 1), r.mean_score_at_deposits == null ? '—' : f(r.mean_score_at_deposits, 3)]);
const ahpRows = (X) => X.ahp.layers.map((n, i) => [n].concat(X.ahp.matrix[i].map((v) => (v >= 1 ? g(v, 3) : `1/${g(1 / v, 3)}`)))
  .concat([f(X.ahp.ahp[i], 3), f(X.ahp.fuzzy1[i], 3)]));
const vwRows = (X) => X.vector_weights.map((v) => [v.label].concat(v.layers.map((n, i) => `${f(v.weights[i], 3)} → ${f(v.effective[i], 3)}`)));
const ent = (X, start) => Object.entries(candW(X, start)).map(([k, v]) => `${k} ${f(v, 2)}`).join(', ');
const vw = (X, label, layer) => { const v = X.vector_weights.find((x) => x.label === label); return v.effective[v.layers.indexOf(layer)]; };

// table numbers (Tables 6.1–6.3 come from the shared demo-data section)
let tn = 3;
const T = () => ++tn;
const tJs = T(), tJa = T(), tJe = T(), tJv = T(), tJc = T();
const tKs = T(), tKa = T(), tKe = T(), tKv = T(), tKc = T();
const tKD = T(), tSet = T(), tIdeal = T();

// knowledge-driven tools on the same demo data (all-deposit ROC AUC of their selected settings)
const KD = [['TOPSIS / VIKOR (this document)', 'topsis'], ['Index overlay', 'index_overlay'], ['Fuzzy logic overlay', 'fuzzy_logic'],
  ['Logistic fuzzification + geometric average', 'logistic_geometric'], ['Boolean logic (condition count)', 'boolean_logic']];
const kdRows = () => KD.filter(([, k]) => C.has(`${k}_carlin`) && C.has(`${k}_curnamona`)).map(([name, k]) => {
  const a = C.load(`${k}_carlin`), b = C.load(`${k}_curnamona`);
  return [name, f(a.reference.curves.training.auc), f(b.reference.curves.training.auc),
    `${f(a.reference.curves.training.pa_prediction, 0)} / ${f(b.reference.curves.training.pa_prediction, 0)}`];
});

const selTxt = (X, key, dflt) => (X.selected[key] === undefined ? dflt : X.selected[key]);
const methodTxt = (X) => (selTxt(X, 'method', 'topsis') === 'topsis' ? 'TOPSIS — relative closeness to the ideal' : 'VIKOR — compromise ranking');
const normTxt = (X) => (selTxt(X, 'method', 'topsis') === 'vikor' ? '— (VIKOR)' : selTxt(X, 'normalisation', 'vector') === 'vector' ? 'vector x / √Σx² (classic TOPSIS)' : 'none — scores 0 … 1');
const wTxt = (X) => ({ equal: 'equal', ahp: 'AHP pairwise comparison (Saaty)', fuzzy_ahp: 'fuzzy AHP', entropy: 'Shannon entropy' }[selTxt(X, 'weighting', 'equal')]);

const setupTable = (cap) => ['table', {
  caption: cap, head: ['Dialog field', 'Carlin (Au)', 'Curnamona (REE)'], widths: [2626, 3200, 3200], rows: [
    ['Project (*Project ▸ Open demo project*)', 'Carlin-type Au, Nevada (USGS)', 'REE, Curnamona, South Australia'],
    ['Study area mask / deposits (validation) / absence points', 'study_mask / deposits / all background cells', 'study_mask / deposits / all background cells'],
    ['Evidence layers (ticked): scoring; scores / range', ev(J).map((r) => `${r.layer}: ${ruleTxt(r.rule)}; ${specTxt(r)}`).join(' | '),
      ev(K).map((r) => `${r.layer}: ${ruleTxt(r.rule)}; ${specTxt(r)}`).join(' | ')],
    ['Weights', wTxt(J), wTxt(K)],
    ['Pairwise matrix (AHP)', `Table 6.${tJa} (CR ${f(J.ahp.cr, 3)})`, '— (equal weights)'],
    ['Missing evidence', 'use the layers with data (weights renormalised)', 'use the layers with data (weights renormalised)'],
    ['Method', methodTxt(J), methodTxt(K)],
    ['Normalisation (TOPSIS) / strategy weight v (VIKOR)', `${normTxt(J)} / —`, `${normTxt(K)} / —`],
    ['Outputs / prefix', 'topsis (+ optional D⁺ / D⁻ and score maps) / topsis', 'topsis / topsis'],
    ['Result (all deposits)', `ROC AUC ${f(auc(J))} (95 % CI ${f(tr(J).auc_lo)}–${f(tr(J).auc_hi)}); P–A ${f(tr(J).pa_prediction, 0)} % of the deposits in ${f(tr(J).pa_area, 0)} % of the area`,
      `ROC AUC ${f(auc(K))} (95 % CI ${f(tr(K).auc_lo)}–${f(tr(K).auc_hi)}); P–A ${f(tr(K).pa_prediction, 0)} % / ${f(tr(K).pa_area, 0)} %`],
  ] }];

module.exports = {
  file: 'topsis.docx',
  code: 'TOPSIS',
  short: 'TOPSIS / VIKOR',
  title: 'TOPSIS and VIKOR Multi-Criteria Decision Making for Mineral Prospectivity Mapping',
  subtitle: 'Knowledge-driven method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Multi-criteria decision making (MCDM) treats prospectivity mapping as a ranking problem: every grid cell is an alternative, every evidence map a criterion, and the cells are ranked by how well they satisfy all criteria together. TOPSIS — the Technique for Order of Preference by Similarity to Ideal Solution (Hwang and Yoon, 1981) — ranks the alternatives by their relative closeness to an ideal solution; VIKOR (Opricovic, 1998; Opricovic and Tzeng, 2004) by a compromise between the total and the largest shortfall from the ideal. TOPSIS, usually with AHP weights, is one of the popular knowledge-driven MCDM methods for porphyry-Cu and other targeting studies (Pazand et al., 2012; Pazand and Hezarkhani, 2015), also with Shannon-entropy or hybrid AHP–entropy weights (Feizi et al., 2017). MPM Studio implements both methods in one tool: *Knowledge-driven ▸ MCDM ▸ TOPSIS / VIKOR (AHP / Fuzzy AHP / entropy weights)*.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4300, 4726], rows: [
      ['`python/logic/knowledge_driven/topsis/model.py`', 'TOPSIS, VIKOR, Shannon-entropy weights, missing evidence, tables'],
      ['`python/logic/knowledge_driven/topsis/plugin.py`, `dialogs.py`', 'Registration, outputs, dialog (the index-overlay evidence table and AHP matrix plus the MCDM options)'],
      ['`python/logic/knowledge_driven/index_overlay/model.py`', 'Shared: scoring rules, AHP and fuzzy-AHP weights'],
      ['`python/tests/test_topsis.py`', 'Textbook TOPSIS and VIKOR steps, entropy weights by hand, missing data, Carlin models'],
    ] }],
    ['h2', '1.1 Place among the knowledge-driven tools'],
    ['p', 'Published comparisons give TOPSIS mixed results against index overlay, fuzzy logic and fuzzy AHP, so it was added under the condition that it be competitive on the demo data; Section 6 shows that it is (Carlin better, Curnamona equal to the index overlay). VIKOR is offered as an option of the same tool rather than a separate menu entry: with the strategy weight v = 1 it ranks the cells exactly like the index overlay (Section 3.3), and v < 1 was never better on the demo data.'],
    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Cells as alternatives, evidence maps as criteria'],
    ['p', 'The criterion values are the expert scores of the index overlay (0 unfavourable … 1 favourable): a score per class for categorical maps, a linear rescaling between two values in the favourable direction for continuous maps, or a binary condition. All criteria are therefore benefit criteria and categorical geology can be used. The scores express *what* is favourable; the MCDM method decides *how* the criteria are traded against each other.'],
    ['h2', '2.2 TOPSIS'],
    ['p', 'The weighted, normalised criteria place every cell as a point in an n-dimensional space. The positive ideal solution (PIS) takes the best value of every criterion found in the study area, the negative ideal solution (NIS) the worst. A good cell is close to the PIS and far from the NIS; the relative closeness C = D⁻/(D⁺ + D⁻) combines both distances into one 0–1 ranking value (Hwang and Yoon, 1981). Because Euclidean distances square the shortfalls, one large shortfall weighs more than several small ones — TOPSIS is compensatory, but less so than the weighted mean of the index overlay.'],
    ['h2', '2.3 VIKOR'],
    ['p', 'VIKOR measures, for every cell, the group utility S (weighted sum of the normalised shortfalls — the L1 distance to the ideal) and the individual regret R (largest weighted shortfall — the L∞ distance). The compromise index Q mixes both with the strategy weight v: v = 1 is pure group utility (full compensation), v = 0 lets the weakest criterion decide (like a fuzzy AND), v = 0.5 is the "consensus" choice (Opricovic and Tzeng, 2004). The acceptable-advantage and stability conditions of VIKOR are meant for choosing among a few alternatives and are not used for maps.'],
    ['h2', '2.4 Weights'],
    ['p', 'Weights can be equal, custom, derived from an AHP or fuzzy-AHP pairwise-comparison matrix (as in the index overlay), or computed by **Shannon entropy** (Shannon, 1948): a criterion whose scores are spread unevenly over the cells carries more information and receives a larger weight. Entropy weights are objective but blind to the deposits: they reward contrast, not the link to mineralisation (Section 6).'],
    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 TOPSIS'],
    ['eq', 'r_{ij}=\\frac{x_{ij}}{\\sqrt{\\sum_{i}{x_{ij}^{2}}}},      v_{ij}=w_{j}r_{ij},      A_{j}^{+}=\\max_{i}v_{ij},      A_{j}^{-}=\\min_{i}v_{ij}', 'tp-norm'],
    ['eq', 'D_{i}^{+}=\\sqrt{\\sum_{j}{(v_{ij}-A_{j}^{+})^{2}}},      D_{i}^{-}=\\sqrt{\\sum_{j}{(v_{ij}-A_{j}^{-})^{2}}},      C_{i}=\\frac{D_{i}^{-}}{D_{i}^{+}+D_{i}^{-}}', 'tp-c'],
    ['p', 'With "no normalisation" $r_{ij}=x_{ij}$ (the scores are already on 0–1). Vector normalisation divides criterion j by $\\|x_{j}\\|=\\sqrt{\\sum_{i}{x_{ij}^{2}}}$, so its effective weight becomes $w_{j}/\\|x_{j}\\|$ (normalised): a layer that is favourable almost everywhere (large norm) loses influence, a layer with rare high scores gains it. $C=1$ at a cell that equals the PIS, $C=0$ at the NIS.'],
    ['h2', '3.2 VIKOR'],
    ['eq', 'S_{i}=\\sum_{j}{w_{j}\\frac{f_{j}^{*}-f_{ij}}{f_{j}^{*}-f_{j}^{-}}},      R_{i}=\\max_{j}w_{j}\\frac{f_{j}^{*}-f_{ij}}{f_{j}^{*}-f_{j}^{-}}', 'vk-sr'],
    ['eq', 'Q_{i}=v\\frac{S_{i}-S^{*}}{S^{-}-S^{*}}+(1-v)\\frac{R_{i}-R^{*}}{R^{-}-R^{*}},      prospectivity=1-Q_{i}', 'vk-q'],
    ['p', 'with $f_{j}^{*}$, $f_{j}^{-}$ the best and worst score of criterion j in the study area and $S^{*}=\\min S$, $S^{-}=\\max S$ (likewise R).'],
    ['h2', '3.3 Relation to the index overlay'],
    ['p', 'If every score spans 0 … 1 in the study area ($f^{*}=1$, $f^{-}=0$), then $S_{i}=\\sum_{j}{w_{j}(1-s_{ij})}=1-I_{i}$ with $I$ the weighted-mean index overlay; with v = 1, Q is a linear function of S and VIKOR ranks the cells exactly like the index overlay. The demo runs confirm this: VIKOR v = 1 and the index overlay with the same scores have the same ROC AUC to four decimals (Tables 6.' + tJs + ' and 6.' + tKs + ').'],
    ['h2', '3.4 Entropy weights and missing evidence'],
    ['eq', 'p_{ij}=\\frac{x_{ij}}{\\sum_{i}{x_{ij}}},      E_{j}=-\\frac{1}{\\ln m}\\sum_{i}{p_{ij}\\ln p_{ij}},      w_{j}=\\frac{1-E_{j}}{\\sum_{k}{(1-E_{k})}}', 'tp-ent'],
    ['p', 'm is the number of study cells with data. A constant layer has E = 1 and weight 0. Missing evidence (default "renormalise"): the weights of a cell are renormalised over the criteria with data, $w_{ij}=w_{j}/\\sum_{k∈obs(i)}{w_{k}}$, and the distances, S and R use only those criteria; "exclude" keeps only cells with every criterion. The PIS, NIS, norms and $f^{*}$, $f^{-}$ are computed over the study area, so the map depends on the extent of the study area (a property of all ideal-point methods).'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Scores** — each ticked layer is converted to 0–1 scores by its rule (classes without a score count as missing).',
      '**Weights** — equal, custom, AHP / fuzzy AHP from the pairwise matrix (weights, λmax, CI and CR shown live), or Shannon entropy [[tp-ent]] from the scores in the study area.',
      '**TOPSIS** — normalise [[tp-norm]], weight, find the PIS and NIS, compute D⁺, D⁻ and the closeness C [[tp-c]]; or **VIKOR** — S, R [[vk-sr]] and Q [[vk-q]], output 1 − Q.',
      '**Validation** — ROC, success rate and P–A plot of all deposits (nothing is trained).',
    ]],
    ['h1', '5 How to use'],
    ['table', { caption: 'Table 5.1. Settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2400, 2300, 4326], rows: [
      ['Scoring / scores / range', 'as the index overlay (classes 0.5; 2nd–98th percentile)', 'Score per class, linear rescaling between two values, or binary condition; the range should keep the anomalies that host deposits distinguishable.'],
      ['Weights', 'equal', 'Equal, custom, AHP, fuzzy AHP or Shannon entropy; with AHP keep CR < 0.1. Entropy only measures the contrast of a layer — check the weights against the deposit model.'],
      ['Pairwise matrix / fuzziness δ', 'all 1 / 1', 'As in the index overlay (upper triangle; reciprocals filled in).'],
      ['Missing evidence', 'renormalise', 'Use the layers with data; "exclude" keeps only cells with every layer.'],
      ['Method', 'TOPSIS', 'TOPSIS (closeness to the ideal) or VIKOR (compromise of group utility and regret).'],
      ['Normalisation (TOPSIS)', 'vector', 'Vector normalisation lowers the influence of layers that are favourable almost everywhere; "none" uses the scores as they are.'],
      ['Strategy weight v (VIKOR)', '0.5', '1 = group utility only (ranks like the index overlay); 0 = the weakest criterion decides.'],
      ['Outputs / prefix', 'topsis or vikor', 'Optional D⁺ / D⁻ (TOPSIS) or S / R / Q (VIKOR) maps and score maps.'],
    ] }],
    ['h1', '6 Demo data: deposit models, results and their geological meaning'],
    ...C.demoData(6, ['Validation protocol (knowledge-driven)', 'all 34 deposit cells (nothing is trained)', 'all 7 occurrences (nothing is trained)']),
    ...C.protocolKD(6, C.MISSING_RENORM),
    ['h2', '6.5 Carlin (Au)'],
    C.searchTable(J, `Table 6.${tJs}. Carlin — configurations (all deposits).`),
    ['table', { caption: `Table 6.${tJa}. Carlin — AHP judgements (row versus column), AHP and fuzzy-AHP weights (δ = 1; λmax = ${f(J.ahp.lambda_max, 3)}, CR = ${f(J.ahp.cr, 3)}).`,
      head: ['Layer'].concat(J.ahp.layers).concat(['AHP w', 'Fuzzy w']), widths: [1726, 1300, 1300, 1300, 1300, 1050, 1050],
      rows: ahpRows(J) }],
    ['table', { caption: `Table 6.${tJe}. Carlin — scores and weights of the selected model.`, head: ['Layer', 'Rule', 'Scores / range', 'Weight', 'Mean score', 'Area with score ≥ 0.5 %', 'Mean score at deposits'], widths: [1500, 1500, 1726, 900, 1000, 1200, 1200], rows: evRows(J) }],
    ['table', { caption: `Table 6.${tJv}. Carlin — nominal → effective weights under vector normalisation (w / ‖x‖, normalised).`, head: ['Configuration'].concat(J.vector_weights[0].layers), widths: [2626, 1600, 1600, 1600, 1600], rows: vwRows(J) }],
    C.curvesTable(J, `Table 6.${tJc}. Carlin — validation with all deposits.`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — TOPSIS relative closeness with the known deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success rate, (c) P–A plot, (d) layer weights.'],
    ['p', `Selected: **${J.selected.label}**, ROC AUC **${f(auc(J))}** (95 % CI ${f(tr(J).auc_lo)}–${f(tr(J).auc_hi)}); at the P–A intersection ${f(tr(J).pa_prediction, 0)} % of the deposits lie in ${f(tr(J).pa_area, 0)} % of the area. Three effects build up this result:`],
    ['bullets', [
      `**Vector normalisation** (${f(cand(J, 'TOPSIS, vector normalisation, equal weights, 2'))} against ${f(cand(J, 'TOPSIS, no normalisation'))} without it). The distance to faults is favourable almost everywhere (in the selected model ${f(ev(J)[1].favourable_area_pct, 0)} % of the area has a fault score ≥ 0.5, mean ${f(ev(J)[1].mean_score, 2)}, at the deposits ${f(ev(J)[1].mean_score_at_deposits, 2)}); its large norm cuts its effective weight from 0.25 to ${f(vw(J, 'equal weights, 2–98 % ranges', 'dist_faults'), 2)}, while As and Sb, high only in their halos, gain (Table 6.${tJv}). The normalisation thus removes influence from the proxy that discriminates least; the fuzzy weights-of-evidence model of the same data likewise gives the faults the coefficient β = 0 once host rock and Sb are known.`,
      `**Full min–max ranges** (${f(cand(J, 'TOPSIS, vector normalisation, equal weights, min'))}). As in the index overlay, six of the 34 deposits lie above the 98th percentile of Sb; the full range keeps ranking the strongest Sb anomalies instead of tying them at the score 1.`,
      `**AHP weights** (${f(auc(J))} with min–max ranges). The judgements rank host and Sb first (${f(J.ahp.ahp[0], 3)} each) and faults last (${f(J.ahp.ahp[1], 3)}); together with the normalisation the faults keep only ${f(vw(J, 'AHP weights, min–max ranges', 'dist_faults'), 2)} of the effective weight. The map is then driven by the reactive host (the chemical trap) and the Sb–As halo (the hydrothermal footprint) — the two components that define Carlin-type gold.`,
    ]],
    ['p', `Entropy weights put most of the weight on the host map (${ent(J, 'TOPSIS, vector normalisation, entropy weights')}; with min–max ranges ${ent(J, 'TOPSIS, vector normalisation, entropy weights, min')}): the binary host layer has the most uneven distribution, not necessarily the strongest link to gold. They helped with the 2–98 % range (${f(cand(J, 'TOPSIS, vector normalisation, entropy weights'))}) and hurt with min–max ranges (${f(cand(J, 'TOPSIS, vector normalisation, entropy weights, min'))}), where the host weight of 0.9 suppresses the Sb–As halo. VIKOR: v = 1 reproduces the index overlay (${f(cand(J, 'VIKOR, v = 1'))}); v = 0.5 (${f(cand(J, 'VIKOR, v = 0.5, equal weights'))}) and especially v = 0 (${f(cand(J, 'VIKOR, v = 0 '))}) are worse, because several deposits are weak in one proxy (a few lie outside the mapped host, many have only moderate As) and the regret term ranks them by that weakest proxy.`],
    ['h2', '6.6 Curnamona (REE)'],
    C.searchTable(K, `Table 6.${tKs}. Curnamona — configurations (all occurrences).`),
    ['table', { caption: `Table 6.${tKa}. Curnamona — AHP judgements, AHP and fuzzy-AHP weights (δ = 1; λmax = ${f(K.ahp.lambda_max, 3)}, CR = ${f(K.ahp.cr, 3)}).`,
      head: ['Layer'].concat(K.ahp.layers.map((n) => n.replace('geochem_nb_log_idw', 'nb_log').replace('aster_silica_index', 'silica'))).concat(['AHP w', 'Fuzzy w']),
      widths: [1926, 1250, 1250, 1250, 1250, 1050, 1050], rows: ahpRows(K) }],
    ['table', { caption: `Table 6.${tKe}. Curnamona — scores and weights of the selected model.`, head: ['Layer', 'Rule', 'Scores / range', 'Weight', 'Mean score', 'Area with score ≥ 0.5 %', 'Mean score at deposits'], widths: [1700, 1500, 1526, 900, 1000, 1200, 1200], rows: evRows(K) }],
    ['table', { caption: `Table 6.${tKv}. Curnamona — nominal → effective weights under vector normalisation (w / ‖x‖, normalised).`, head: ['Configuration'].concat(K.vector_weights[0].layers.map((n) => n.replace('geochem_nb_log_idw', 'nb_log').replace('aster_silica_index', 'silica'))), widths: [2626, 1600, 1600, 1600, 1600], rows: vwRows(K) }],
    C.curvesTable(K, `Table 6.${tKc}. Curnamona — validation with all occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — TOPSIS relative closeness with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) layer weights.'],
    ['p', `Selected: **${K.selected.label}**, ROC AUC **${f(auc(K))}** (95 % CI ${f(tr(K).auc_lo)}–${f(tr(K).auc_hi)}; index overlay ${f(ioAuc(K))}). Here vector normalisation is harmful (${f(cand(K, 'TOPSIS, vector normalisation, equal weights, 2'))}): the radiometric Th and U scores are low over most of the area (mean ${f(ev(K)[0].mean_score, 2)} and ${f(ev(K)[1].mean_score, 2)}), so their small norms raise their effective weights to ${f(vw(K, 'equal weights, 2–98 % ranges', 'rad_th'), 2)} and ${f(vw(K, 'equal weights, 2–98 % ranges', 'rad_u'), 2)}, while Nb and the silica index fall to ${f(vw(K, 'equal weights, 2–98 % ranges', 'geochem_nb_log_idw'), 2)} and ${f(vw(K, 'equal weights, 2–98 % ranges', 'aster_silica_index'), 2)} (Table 6.${tKv}). This is the same shift as the AHP judgements (Th and U 70 %), which also lowered the index overlay (${f(io(K).search.find((s) => s.label.startsWith('multi-class, AHP')).mean_test_auc)}). Th and U are clearly higher at the occurrences than on average, but the occurrences have moderate rather than maximal radiometric scores (mean ${f(ev(K)[0].mean_score_at_deposits, 2)} and ${f(ev(K)[1].mean_score_at_deposits, 2)}): the radiometric maxima of the source terrain lie mostly elsewhere, and when Th and U dominate, these barren highs outrank the occurrences. Nb and the silica index are only moderately but consistently higher at the occurrences (${f(ev(K)[2].mean_score_at_deposits, 2)} against ${f(ev(K)[2].mean_score, 2)}, ${f(ev(K)[3].mean_score_at_deposits, 2)} against ${f(ev(K)[3].mean_score, 2)}) and stabilise the ranking when they keep their share. Without normalisation and with equal weights the four mineral-system proxies count equally and TOPSIS equals the index overlay; entropy weights (${ent(K, 'TOPSIS, vector normalisation, entropy weights')}) push even more weight onto the skewed radiometric layers (${f(cand(K, 'TOPSIS, vector normalisation, entropy weights'))}). VIKOR v = 1 again equals the index overlay (${f(cand(K, 'VIKOR, v = 1'))}); v = 0 is the weakest (${f(cand(K, 'VIKOR, v = 0 '))}).`],
    ['h2', '6.7 Comparison with the other knowledge-driven tools'],
    ['table', { caption: `Table 6.${tKD}. Knowledge-driven tools on the demo data (selected settings; ROC AUC and P–A deposits % with all deposits).`,
      head: ['Tool', 'Carlin AUC', 'Curnamona AUC', 'P–A deposits % (Carlin / Curnamona)'], widths: [3826, 1500, 1700, 2000], rows: kdRows() }],
    ['p', `TOPSIS is the most accurate knowledge-driven tool on Carlin (${f(auc(J))} against ${f(ioAuc(J))} for the index overlay) and equal to the index overlay on Curnamona, where the fuzzy overlay is marginally better. The gain on Carlin is within the 95 % confidence interval of the AUC (${f(tr(J).auc_lo)}–${f(tr(J).auc_hi)}) and the setting was chosen among ${J.search.length} configurations with the same deposits, so it should be read as "at least as good as the index overlay", not as a decisive improvement.`],
    ['h2', '6.8 Recommended setup for the demo projects'],
    ['p', 'The settings below reproduce the selected results from the TOPSIS / VIKOR dialog. Linear ranges are typed as "low, high"; the dialog proposes the 2nd–98th percentile, so the Carlin min–max ranges must be typed. Section 6.9 explains the reasoning.'],
    setupTable(`Table 6.${tSet}. TOPSIS / VIKOR dialog settings for the demo projects.`),
    ['h2', '6.9 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${tIdeal}. Ideal settings of the demo projects.`,
      head: ['Element', 'Carlin (Au)', 'Curnamona (REE)', 'Geological and ore-forming reasoning'],
      widths: [1500, 1800, 1800, 3926],
      rows: [
        ['Method', 'TOPSIS', 'TOPSIS', `Both deposit types are captured by a compensatory combination: Carlin deposits are weak in one proxy at some sites, the Curnamona occurrences are not all high in Th and U. VIKOR with v < 1 (regret) penalised exactly those sites (Carlin v = 0: ${f(cand(J, 'VIKOR, v = 0 '))}; Curnamona ${f(cand(K, 'VIKOR, v = 0 '))}).`],
        ['Scores', 'host 1 / other 0; linear As, Sb (high), faults (low)', 'linear Th, U, Nb (high), silica index (low)', 'Reactive silty carbonate is the chemical trap of Carlin-type gold; As–Sb halos and faults are the hydrothermal footprint and pathway. In Curnamona Th and U mark the granitic source, Nb the co-enrichment, low silica index the exposure.'],
        ['Range', J.selected.robust === false ? 'data min–max' : '2–98 %', K.selected.robust === false ? 'data min–max' : '2–98 %', 'The range must keep the strongest anomalies distinguishable: the richest Sb values (six Carlin deposits above the 98th percentile) must not saturate, while in Curnamona a few extreme radiometric values over granite cores would compress the contrast of the rest.'],
        ['Normalisation', normTxt(J), normTxt(K), `Vector normalisation lowers the weight of layers that are favourable almost everywhere (Carlin faults → ${f(vw(J, 'AHP weights, min–max ranges', 'dist_faults'), 2)}) and raises that of layers with rare high scores — useful when the widespread layer is the weak proxy (Carlin), harmful when the occurrences have moderate, not maximal, scores in the rare-high layers (Curnamona Th, U).`],
        ['Weights', wTxt(J), wTxt(K), 'AHP when the expert can rank the processes with confidence (Carlin: trap and Sb halo above As and faults); equal weights when the proxies are independent and none dominates (Curnamona). Entropy weights follow the shape of the score distribution, not the deposit model, and were not selected.'],
        ['Missing evidence', 'renormalise', 'renormalise', 'Gaps in the host map (unmapped units) must not erase the evidence of the other maps.'],
      ],
    }],
    ['h2', '6.10 Discussion'],
    ['bullets', [
      `**Accuracy.** ROC AUC of all deposits: Carlin ${f(auc(J))}, Curnamona ${f(auc(K))} — at least as good as the index overlay on both projects.`,
      '**What TOPSIS adds.** The Euclidean distances to the best and worst cells and, with vector normalisation, an automatic down-weighting of layers that are favourable almost everywhere. Whether that down-weighting is right is a geological question — inspect the effective weights.',
      '**VIKOR.** Useful to explore the trade-off between compensation (v = 1, identical to the index overlay) and the weakest-criterion view (v = 0); on the demo data compensation was better.',
      '**Limits.** Relative ranking only (no probability); the ideal solutions and norms depend on the study-area extent, so maps of different extents are not comparable; compensatory like the index overlay — use Boolean or fuzzy AND operators, or a permissive-tract gate, where a component is strictly required.',
    ]],
    ['h1', '7 References'],
    ['refs', R('bonham1994', 'buckley1985', 'carranza2008', 'chung2003', 'cline2005', 'deepseam', 'delong1988',
      'elburg2012', 'fawcett2006', 'feizi2017', 'hofstra2000', 'hwang1981', 'muntean2011', 'neumann2000', 'opricovic1998',
      'opricovic2004', 'pazand2012', 'pazand2015', 'raines2001', 'saaty1980', 'shannon1948', 'yousefi2015')],
  ],
};
