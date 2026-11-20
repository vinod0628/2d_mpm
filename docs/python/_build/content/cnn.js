const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('cnn_carlin');
const K = C.load('cnn_curnamona');
const P = (X) => X.reference.tables.params;
const st = (X) => X.reference.tables.stats;
const imp = (X) => X.reference.tables.importance;
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const cand = (X, lab) => { const r = X.search.find((s) => s.label.startsWith(lab)); return r ? r.mean_test_auc : null; };
let tn = 3;
const T = () => ++tn;
const impRows = (X) => imp(X).map((r) => [String(r.rank), r.layer, String(r.columns), pm(r.permutation_auc_drop, r.permutation_sd, 3)]);
const svRows = (X) => X.reference.tables.sample_validation.map((r) => [r.item, r.value]);
const sumRows = (X) => X.reference.tables.summary.map((r) => [r.item, r.value]);
const win = (X) => P(X).patch * X.dataset_info.cell_m;

module.exports = {
  file: 'cnn.docx',
  code: 'CNN',
  short: 'Convolutional neural network',
  title: 'Convolutional Neural Network (CNN) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven deep-learning method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Convolutional neural networks (LeCun et al., 1998) learn spatial patterns from images. In mineral prospectivity mapping a patch-based CNN represents every cell by the window of all evidence layers around it, so the network can use the spatial context of a deposit — the shape of an anomaly, the proximity and orientation of contacts and structures — and not only the values at the cell (Sun et al., 2020). MPM Studio (*Data-driven ▸ Machine learning ▸ Convolutional Neural Network*) uses TensorFlow / Keras, augments the few deposit patches by their rotations and reflections, chooses the number of epochs by cross-validation and averages an ensemble of networks.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4000, 5026], rows: [
      ['`python/logic/data_driven/cnn/model.py`', 'Patch grid, D4 augmentation, Keras network, cross-validated epochs, ensemble, importance'],
      ['`python/logic/data_driven/cnn/plugin.py`, `dialogs.py`', 'Registration, outputs, dialogs (window size shown in map units)'],
      ['`python/tests/test_cnn.py`', 'Augmentation, exact patch extraction, spatial signal, Carlin model'],
    ] }],
    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Convolution and spatial context'],
    ['p', 'A convolution layer slides small filters (here 3 × 3 cells) over the input window and produces feature maps that respond to local patterns — gradients, edges, spots — wherever they occur; stacked layers combine them into larger patterns, pooling summarises them. The network therefore learns which spatial configurations of the evidence surround deposits, e.g. a geochemical anomaly at the margin of a favourable unit next to a fault.'],
    ['h2', '2.2 Few deposits: augmentation, dropout, small networks'],
    ['p', 'CNNs have many weights and MPM has few deposits. Three measures keep the network from memorising them: rotations and reflections of every training patch (geological patterns have no preferred orientation at the patch scale, so the 8 symmetries of the square are equally valid examples), dropout of the dense layer (Srivastava et al., 2014) and a deliberately small architecture. The number of epochs is chosen on held-out deposits (early stopping).'],
    ['h1', '3 Mathematical basis'],
    ['eq', 'z_{c}^{(l)}(u,v)=\\rm{ReLU}(b_{c}+\\sum_{c\'}{\\sum_{s,t=-1}^{1}{w_{c c\' s t}z_{c\'}^{(l-1)}(u+s,v+t)}})', 'cnn-conv'],
    ['p', 'for every output channel $c$ and position $(u,v)$ of the $k×k$ window; max pooling over 2 × 2 blocks, a dense ReLU layer with dropout and a sigmoid output $P$ follow. Training minimises the binary cross-entropy with Adam (Kingma and Ba, 2015) on the augmented samples $\\{g(\\bf{X}_{n}):g∈D_{4}\\}$, the 8 rotations / reflections. Number of epochs: $E^{*}$ = median over the folds of the epoch with the best held-out AUC; ensemble mean and s.d. over networks and random barren samples as for the MLP.'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Patch grid** — z-scored evidence of the whole grid (cells outside the study area = 0 = mean), zero-padded by half a window.',
      '**Training samples** — windows of training deposits and random non-deposit cells; ×8 augmentation.',
      '**Epochs** — k-fold networks with EarlyStopping on the held-out AUC (best weights restored); $E^{*}$.',
      '**Ensemble** — networks with different seeds trained $E^{*}$ epochs on all samples; prediction of all study cells in chunks of 50 000 windows.',
      '**Importance** — windows of one layer exchanged between held-out samples; drop of the CV AUC.',
    ]],
    ['h1', '5 How to use'],
    ['table', { caption: 'Table 5.1. Model options.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2400, 1700, 4926], rows: [
      ['Patch size', '9 cells', 'Spatial context; the dialog shows it in map units. Should cover the footprint of a deposit and its halo.'],
      ['Convolution filters / dense neurons', '8, 16 / 16', 'Small by design (few deposits).'],
      ['Dropout / learning rate', '0.5 / 0.0003', 'Regularisation and step size; larger values over-fitted within a few epochs on Carlin.'],
      ['Batch / max. epochs / early stopping', '32 / 200 / 20', ''],
      ['Augmentation', 'on', '8 rotations / reflections of every training patch.'],
      ['Networks per sample / repeats / folds', '2 / 5 / 3', 'Ensemble and cross-validation.'],
    ] }],
    ['p', 'TensorFlow uses an NVIDIA GPU automatically when it can see one; TensorFlow ≥ 2.11 has no GPU support on native Windows, so there the CPU is used.'],
    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),
    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, `Table 6.${T()}. Evidence of the Carlin CNN.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (2 samples × 1 network, spatial 70 / 30 splits).`),
    ['p', `The selected setting is **${J.selected.label}**. Over 10 splits it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** with a training AUC of ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)}.`],
    C.robustTable(J, `Table 6.${T()}. Carlin — robustness over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42).`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — CNN prospectivity score with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) learning curve.'],
    ['table', { caption: `Table 6.${T()}. Carlin — importance (CV AUC drop).`, head: ['Rank', 'Layer', 'Channels', 'CV AUC drop'], widths: [900, 3726, 1400, 3000], rows: impRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — model summary.`, head: ['Item', 'Value'], widths: [4026, 5000], rows: sumRows(J) }],
    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, `Table 6.${T()}. Evidence of the Curnamona CNN.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — CNN prospectivity score with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) learning curve.'],
    ['p', `Mean LODO AUC of the selected setting: **${f(lodo(K))}**. With seven occurrences (56 augmented positive windows) the CNN has the least data of all methods; its result should be read together with the simpler models in the companion documents.`],
    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal CNN parameters for the demo projects.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1600, 1400, 1400, 4626],
      rows: [
        ['Patch size', `${P(J).patch} cells = ${int(win(J))} m`, `${P(K).patch} cells = ${int(win(K))} m`, `The window must contain the deposit and its alteration / geochemical halo and the nearby controlling structure, but not unrelated geology. Carlin: patch 5 ${f(cand(J, 'patch 5'))}, 9 ${f(cand(J, 'patch 9, 8-16'))}, 15 ${f(cand(J, 'patch 15'))} held-out AUC — a window of a few km matches the reach of Carlin-type fluid flow from faults into the host beds.`],
        ['Augmentation', P(J).augment ? 'on' : 'off', P(K).augment ? 'on' : 'off', `Fault and contact orientations vary between deposits; rotations / reflections teach the network the configuration, not the azimuth. Carlin without augmentation: ${f(cand(J, 'patch 9, no augmentation'))}.`],
        ['Network size', '8 / 16 filters, 16 neurons', '8 / 16 filters, 16 neurons', 'Tens of deposits support only a few thousand weights; larger networks over-fitted within 2–5 epochs.'],
        ['Non-deposit ratio', `1 : ${f(P(J).neg_ratio, 0)}`, `1 : ${f(P(K).neg_ratio, 0)}`, 'Barren windows show the varied geology without deposits; a few per deposit balance the classes in every batch.'],
        ['Epochs', `${st(J).epochs}`, `${st(K).epochs}`, 'Chosen on held-out deposits; the network should learn the regional deposit setting, not the individual windows.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** Mean test AUC on Carlin ${f(J.robust_summary.test_auc_mean)}; mean LODO AUC on Curnamona ${f(lodo(K))}.`,
      '**Context versus data.** The CNN can use spatial context that pixel-based methods ignore, but needs many examples; with tens of deposits its advantage is limited and regularisation is essential.',
      '**Cost.** Training and prediction are the slowest of all methods (TensorFlow on the CPU under Windows).',
    ]],
    ['h1', '7 References'],
    ['refs', R('carranza2008', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006', 'hofstra2000',
      'kingma2015', 'lecun1998', 'muntean2011', 'neumann2000', 'raines2001', 'roberts2017', 'srivastava2014', 'stevens2004',
      'sun2020', 'yousefi2015', 'zuo2020')],
  ],
};
