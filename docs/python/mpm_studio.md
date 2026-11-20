# MPM Studio — 2D mineral prospectivity mapping application

PyQt5 desktop application (Python 3.10) for knowledge- and data-driven MPM.
Status: **framework + data tools + Weights of Evidence (WofE) + Logistic Regression (LR) + Random Forest (RF) + Support Vector Machine (SVM) + XGBoost + Relevance Vector Machine (RVM) + Neural Network (MLP) + Extreme Learning Machine (ELM) + CNN + Bayesian Network + fuzzy / Boolean / logistic-geometric overlays + ROC/AUC validation**. Further methods
plug into the same framework one by one (see §6).

Version 1.0.0. Windows release (stand-alone exe, no Python needed): `python\build_release.bat` → `python\distribute\release\MPM_Studio_1.0.0_win64.zip`, see [how_to_build_python_app.md](how_to_build_python_app.md).

## 1. Start

```bat
python\run_mpm.bat                            :: or: .venv\Scripts\python.exe python\main.py
python\run_mpm.bat D:\mpm\my_project          :: open a project folder (or its project.mpm.json)
```

At start-up no project is open: only *New project*, *Open project*, *Recent projects*, *Open demo project*, the view tools, Settings and Help are
available. Everything else (adding data, data tools, methods, validation, saving) is enabled once a project is open.

First time only, build the analysis-ready demo layers (≈40 s), or use **Project ▸ Open demo project ▸ Rebuild demo data…**:

```bat
cd python
..\.venv\Scripts\python.exe -m utils.prepare_demo_data
```

## 2. Window layout

| Area | Content |
|---|---|
| Menu bar | Project · View · Data · Knowledge-driven · Data-driven · Hybrid · Validation · Settings · Help |
| Toolbar | open/save project · add raster/vector · pan, **measure**, box zoom, zoom ±, full extent, identify · reclassify, align, distance · WofE · ROC/AUC (LR: Data-driven ▸ Statistical menu) |
| Left dock **Layers** (TOC) | checkbox = visible; drag to reorder (top = drawn last); double-click = properties; right-click = zoom to layer, properties, attribute table, move, save as (rasters: GeoTIFF; vectors: **ESRI Shapefile** or GeoPackage), **export legend to PNG** (all legend entries of the current symbology, black text on white, 300 dpi with **10 pt** text — true size when inserted in Word at 100 %; the width follows the longest text), remove. Below: details of the selected layer (size, cell size, CRS, statistics, classes) |
| Large projects | rasters are read **lazily**: opening a project reads only their headers; the visible rasters are read by background threads (in parallel) and drawn as they arrive — the window stays responsive. Hidden rasters are read when they are shown or used. Distinct values / counts and statistics of a raster are computed once |
| Centre **map view** | mouse wheel = zoom at cursor; **hold the middle button and drag = pan in any tool** (also left-drag in Pan mode) — snapshot panning as in ArcMap/QGIS: the map image follows the cursor, the legend stays fixed, and the map is re-rendered once on release; box zoom, identify; legend of the selected layer (colour bar or class swatches) |
| **Coordinate bar** (below the map) | left: cursor X / Y in the map CRS (decimal degrees when the project CRS is geographic); right: project raster cell size (*Cell 200 m*) and map CRS |
| **Identify** dialog | opens when the Identify tool is active; click the map → top: tree of identified layers (raster value + class label; vector layers expand to the hit features, nearest first), bottom: attributes of the selected item (topmost layer by default; raster cell row/column, centre X/Y and class-table fields). Drag the divider to resize; *Layers* = all / visible / selected layer. Closing it returns to Pan |
| **Measure** tool (ruler, right of Pan, key `M`) | small dialog with **Distance** / **Area** tabs (Distance by default; the last tab used is remembered). Left-click adds points, the last segment follows the mouse in real time (dashed), double-click or right-click finishes, Esc / *Clear* starts over. Distance: segment, total length and the **geodetic azimuth** from the last point to the current point (clockwise from true north: 0° N, 90° E; the grid azimuth is shown when it differs). Area: area (km², ha, m²) and perimeter of the outline (no fill). Projected CRS → planar lengths/areas in metres; geographic CRS → ellipsoidal (geodesic). Closing the dialog, choosing another tool or running any other command clears the temporary graphics |
| Bottom dock **Output** | *Log* (all messages, warnings, errors) |
| Status bar | task message and progress |

Italic layer names = in-memory layers (e.g. from Reclassify / Align / Distance; Clip always writes its file) not yet written to disk; *Save
project* writes them to the project's `data\` folder. The title bar shows `*` while there are unsaved changes, and
*New / Open / Exit* ask to save them.

### Projects (folder based)

A project is a folder. The project file stores layer paths (relative inside the folder), order, visibility, styles
and the map CRS; the folder can be copied or moved as a whole.

```
<project folder>\
├─ project.mpm.json            the project file — exactly one *.mpm.json per project folder
├─ data\                       layers created in the project (saved in-memory layers, tool outputs)
└─ results\
   └─ wofe_1118_153012\        one folder per method run: <method or prefix>_<MMDD>_<HHMMSS>
      ├─ post.tif  post_std.tif  conf.tif        short, ASCII file names (ArcMap / ArcGIS friendly)
      └─ weights_<layer>.csv  summary.csv  pairwise.csv  validation.csv  run_info.txt
```

| Command | Behaviour |
|---|---|
| **New project…** | choose a folder; if it is not empty a warning offers *Choose another folder* / *Continue*. The project is created and saved immediately (`project.mpm.json`, `data\`, `results\`), then *Project properties* opens on the coordinate-system tab |
| **Open project…** | choose a project folder. It must contain exactly one `*.mpm.json` in its root: none or several → warning, nothing is opened |
| **Recent projects** | the last 5 projects created, opened or saved as (newest first; name = project file name, or the folder name for `project.mpm.json`, with the folder path). A moved / deleted project is greyed out *(not found)*; *Clear recent projects* empties the list. The list is a user preference (registry `HKCU\Software\MPM`) |
| **Open demo project** | choose a folder (same warning) — the demo is copied there as a new project, the demo data itself stays untouched |
| **Save project** | no dialog: in-memory layers → `data\`, then the project file is updated |
| **Save project as…** | choose another folder (same warning); the whole project folder is copied there and work continues in the copy |
| **Project properties…** (`Ctrl+Shift+P`) | see below |
| **Method run** | outputs are written immediately to `results\<run_id>\` (not kept in memory), layers are named `<run_id>_<output>` (e.g. `wofe_1118_153012_post`), result tables are saved as CSV and the project file is updated |

### Project properties

| Tab | Content |
|---|---|
| **General** | name, location (*Open folder*), project file, last saved, size on disk, number of method runs, layers (raster / vector / in memory), coordinate system, raster cell size and project grid (columns × rows) |
| **Extent** | **Project grid (shared by all rasters)**: extent X/Y min–max (*Extent from a raster file…*), **raster cell size**, the resulting grid (columns × rows, cell area) and the extent of all layers; table of the raster layers with their cell size, size and whether they are *on the project grid* |
| **Coordinate system** | *Type*: not set · geographic (GCS) · projected UTM · other projected (PCS) · custom. *Datum*: WGS 84, WGS 72, NAD83, NAD27, ETRS89, ED50, OSGB36, RGF93, GDA2020, GDA94, NZGD2000, CGCS2000, Beijing 1954, Xian 1980, Pulkovo 1942, JGD2011, Tokyo, Korea 2000, Indian 1975, SIRGAS 2000, SAD69, Hartebeesthoek94, Arc 1960, Adindan. *UTM zone*: all 120 zones −60 … −1 (60S … 1S) and +1 … +60 (1N … 60N) with their longitude band — the official EPSG code is used when there is one (e.g. WGS 84 / UTM 50N = EPSG:32650, GDA94 zone 54 = MGA EPSG:28354), otherwise UTM on that datum. *Projection*: web / world (Pseudo-Mercator, Mollweide, Robinson …), continental (LAEA Europe, Conus Albers, Australian Albers …), national grids and polar. *Custom*: EPSG code, WKT or PROJ string. *From a data file…* takes the CRS of a raster or vector file. Details: name, units, WKT |

**NoData value** (Extent tab, default **-9999**, editable at any time): written into every raster the MPM methods and tools create (method results, Distance, Clip default) for cells without a value; methods also receive it as `params["nodata"]`. 8-bit class rasters that cannot hold it are written with a wider integer type.

**Project grid.** The project **coordinate system**, **extent** and **raster cell size** (and the cell alignment that
follows from them) are the reference for tools such as *Align raster to grid*; added rasters are not forced onto it.
They can be set while the project has **no layers** (e.g. right after *New project*); once a layer is loaded they
are **read-only**. Unset values are taken from the **first layer** loaded (coordinate system) and the **first raster
layer** (extent and cell size; with a cell size set beforehand the extent is the raster's extent snapped outwards to
multiples of the cell size). Removing all layers makes them editable again. Coordinate systems come from the local
EPSG database (`proj.db` shipped with pyproj) — no internet access (`PROJ_NETWORK=OFF`).

**Adding data** (*Data ▸ Add raster / Add vector*, *Convert … ▸ Add result to the map*):

| Case | What happens |
|---|---|
| any raster | **loaded exactly as stored in the file** — no question, no conversion: coordinate system, cell size, extent and alignment are not checked against the project. A raster whose coordinate system differs from the project's is only noted in the log (it is drawn with its own coordinates). To put a raster on the project grid, use *Data ▸ Align raster to grid…* or *Convert …* |
| vector in another coordinate system | reprojected to the project coordinate system (lossless; the file itself is not changed) |

Methods that combine rasters resample the evidence to the grid of the first evidence layer when the grids differ
(option *Resample evidence to the first layer's grid*); the layers should be in one coordinate system.

### Layer properties (double-click a layer)

Tabs as in ArcMap; the settings are saved in the project file (`symbology` of each layer).

| Tab | Content |
|---|---|
| **General** | name (editable), source file (or *in-memory layer*), type (raster data type and columns × rows / vector geometry and feature count), extent, coordinate system; rasters also cell size, NoData and *Values are classes (categorical)* — the data type used by the methods (nearest-neighbour resampling, class lists) |
| **Symbology** | opacity and a list of renderers on the left — the one selected when OK is pressed is used: |
| | **Single symbol** — every feature with one symbol (points: marker, fill, border colour, size; lines: colour, width; polygons: fill or *no fill*, border colour, border width); rasters: every valid cell in one colour |
| | **Unique values** — vectors: every feature its own symbol, listed with a *label field*; a colour ramp gives the default colours, **double-click a feature** (or select several and *Change symbol of selected features…*) to set marker, fill and border. Rasters: every cell value its own colour (class table `color` column, else the ramp); double-click a row to change a colour, edit the label column to rename a class (up to 1024 values) |
| | **Stretched** (rasters) — colour ramp (optionally inverted) between minimum and maximum; *Min–max* / *2–98 %* stretch buttons; statistics |
| | **Categorized** — classes of an attribute: a vector field, or for rasters the cell value or a class table column (e.g. `host`). *Method*: unique values, equal interval, quantile or **natural breaks (Jenks)** with a number of classes (ranges only for numbers; a numeric field with more than 32 different values defaults to natural breaks — e.g. contour elevations); colour ramp; *Classify* fills the table (symbol, value or range, label, count). Double-click a symbol to change it, double-click a label to rename it, *Remove* drops classes (features / cells without a class are not drawn) |
| **Labels** (vectors) | *Label features on the map*, label field, font, font size, font colour. Labels are placed above the point / on the line / inside the polygon — only in the current view, a label that would overlap another one is left out (as in QGIS / ArcMap; zoom in to see more) and at most 600 labels are drawn at once, so layers with thousands of features stay fast |

The map legend follows the renderer (colour bar for stretched rasters, one entry per class otherwise).

## 3. Data menu (pre-processing)

*Add raster…* (`Ctrl+R`) and *Add vector…* (`Ctrl+Shift+V`) are at the top of the Data menu.


| Dialog | What it does | Code |
|---|---|---|
| Convert to GeoTIFF / GeoPackage | any GDAL raster (incl. ArcInfo grid folders) → compressed GeoTIFF + `.classes.csv` from the attribute table; any OGR vector → GeoPackage; assigns a CRS if missing | `utils/convert.py` |
| Reclassify raster | continuous → classes by equal interval, quantile, natural breaks (Fisher–Jenks), standard deviation or manual breaks, with histogram + class-table preview | `utils/reclass.py` |
| Align raster to grid | reproject/resample onto a template grid (nearest, bilinear, cubic, average, mode) | `utils/raster_tools.align_to_grid` |
| Clip raster by mask | keep cells inside (or outside) the polygons of a mask layer, optionally only features where *field = value*; cell centre or all-touched rule; optional crop to the mask extent (off by default, so the result stays on the project grid); output NoData (default −9999; 8-bit / unsigned integer rasters are widened so the value fits) and compression; default output `results/<input file>_clip.tif`; no resampling, the mask is reprojected to the raster CRS | `utils/raster_tools.clip_by_mask` |
| Distance to features | Euclidean distance to vector features (0 inside polygons, or to outlines) | `utils/raster_tools.distance_to_features` |

The same functions work from the command line (`python -m utils.convert …`, `python -m utils.reclass …`).

**Class tables.** A categorical raster `x.tif` can carry `x.classes.csv` with columns `value,label[,lower,upper,count,color,…]`.
Labels appear in the legend, the TOC info panel, identify and the WofE tables. If `lower/upper` exist (made by
Reclassify), the WofE dialog proposes *cumulative ascending* by default.

## 4. Weights of Evidence (Data-driven ▸ Statistical ▸ Weights of Evidence…)

Parameter dialog:

| Section | Setting | Notes |
|---|---|---|
| Training data | Deposits (point layer), study-area mask (raster), hold-out % for testing, random seed | test deposits are excluded from fitting and reported separately |
| Evidence layers | tick layers; per layer: weight type, generalisation criterion, minimum Stud(C) | *Categorical*: one weight per class. *Cumulative ascending* (≤ threshold; e.g. distance to faults) / *descending* (≥ threshold; e.g. geochemistry) → binary pattern at the threshold with max C among Stud(C) ≥ minimum (or max Stud(C)) |

Evidence must be **integer (class) rasters**, as in ArcSDM; real-valued layers are disabled in the table (shown
"—"). Nothing is classified automatically: the user decides how to classify a real-valued layer (method, number of
classes) with *Data ▸ Reclassify raster*, which writes an integer class raster that can then be ticked. There is no class
limit for the cumulative types (every class value is a candidate threshold, e.g. distance in whole metres); the
categorical type allows up to 1,000 classes. For layers with many values the dialog proposes *maximum studentized
contrast*, because *maximum contrast* over many fine thresholds tends to pick tiny patterns holding 1–2 deposits.
A cumulative pattern without deposits is never selected.
| Model options | unit cell area (km², default 1), missing evidence (weight 0 / exclude), classes without deposits (weight 0 / 0.5-deposit correction), resample to first grid, outputs, name prefix | |

Outputs added to the map: `<prefix>_post`, `_post_std` (due to the weights), `_conf` (P / total s(P)), optional
generalised weight rasters; when evidence is missing inside the study area (missing → weight 0) also `_md_var`
(variance due to missing data) and `_total_std` = √(s²(P) + Var_MD). Var_MD follows Bonham-Carter (1994, ch. 9) as
implemented in ArcSDM `missingdatavar_func.py`: for each class j of a layer with missing cells,
Var_MD += (P_j − P)²·A_j/A with O_j = O·exp(W_j).

Results dialog: metric cards (training deposits, prior, CI ratio, AUC train/test), one tab per evidence layer with the
full weights table (N, deposits, W+, s(W+), W−, s(W−), C, s(C), Stud(C), selected threshold highlighted) and the
generalised weights, a conditional-independence tab (overall N(D)/T ratio, Agterberg–Cheng one-tailed test, pairwise
χ² on deposit cells) and *Export tables (Excel)*. **ROC / AUC…** opens the validation dialog.

Formulas follow Bonham-Carter (1994). Weights were verified against `eis_toolkit` 1.1.6 on the Carlin data
(`python/tests/test_wofe.py`). The conditional-independence tab also reports ArcSDM's figures: probability of
conditional dependence = 100·Φ((T − n)/s(T)) and overall CI %.

**Comparison with ArcSDM** (GTK, `ArcSDM-master`: `calculateweights.py`, `calculateresponse.py`,
`agterbergchengci.py`) — same W+/W−/C/s(C) formulas. Deliberate differences, where we judge our rule statistically
better:

| Topic | ArcSDM | MPM Studio | Why |
|---|---|---|---|
| training points | every point counts (two in one cell count twice) | one per cell | avoids pseudo-replication (ArcSDM recommends thinning to one point per unit cell) |
| class with 0 or all deposits | ad-hoc: 0 → weights 0; all → db − 0.99; areas patched with 0.99 / +1 | weight 0, or Haldane 0.5-deposit correction | standard continuity correction |
| no threshold with Stud(C) ≥ minimum | table left incomplete (run fails) | largest Stud(C) used, with a warning | the run always completes |
| s(P) with one evidence layer | W_STD itself (std of a weight) | √(P²(1/n + s²(W))) for any number of layers | consistent units (probability) |
| confidence | P / s(P) without the missing-data variance (its comment says total) | P / total s(P) | includes all uncertainty |
| Agterberg–Cheng s(T) | summed by distinct s(P) value | summed by unique condition (as in the paper) | exact grouping |
| missing-data variance | identifies missing cells by weight = 0 | true missing-data mask | known classes with weight 0 are not "missing" |

### Training points (all data-driven methods) — `python/logic/common/training.py`

| Step | Setting | Logic | ArcSDM equivalent |
|---|---|---|---|
| deposits → cells | — | points inside the study area / evidence grid; one per cell | (ArcSDM counts every point) |
| thinning | *Thin deposits (min. distance)*, default off | greedy in input order: a deposit is kept only if ≥ d from every kept deposit; d = √(unit cell / π) ≈ 564 m for 1 km² | *Thinning* tool |
| train / test split | *Hold out for testing* (0–50 %, default 20 %; 0 = all deposits train), seed, method | **spatially stratified** (default): deposits ordered along a Hilbert curve, cut into n_test consecutive runs, one random test deposit per run → test and training deposits both cover the whole area (spatially balanced sampling, Stevens & Olsen 2004). **random**: shuffle, first x % train | *Splitting* tool (random) |
| non-deposit samples (sampled LR, later ML methods) | samples per deposit, min. distance from deposits, **min. spacing between samples** (auto = ½·√(available area / n), 0 = none) | candidates = study area ∩ all evidence known ∖ buffer around every known deposit; dart throwing in random order, rejecting samples closer than the spacing | *Create Random Points* (constraining area + rasters, buffer around points, minimum allowed distance) |

Every data-driven run also adds the deposits it used as two point layers, written to `results\<run_id>\` as
**Shapefiles** (`training_points.shp`, `test_points.shp` with `.prj` and UTF-8 `.cpg`, so they open in ArcMap):
`<run_id>_training_points` (green circles) and, when deposits are held out, `<run_id>_test_points` (blue triangles).
They are the original deposit features with all their attributes plus a `split` column; several deposits in one
cell share that cell's set, and deposits outside the study area or removed by thinning are in neither layer.

The dialogs hold out 20 % by default, which gives an independent test AUC. With few deposits (< ~30) consider 0 %
(all for training), because the weights / coefficients become unstable otherwise.

## 4b. Logistic Regression (Data-driven ▸ Statistical ▸ Logistic Regression…)

Model (Agterberg 1974; Agterberg et al. 1993; Bonham-Carter 1994): **logit P = β0 + Σ βj·xj**, coefficients estimated
jointly, so correlated evidence layers are not double-counted (no conditional-independence assumption). Code:
`python/logic/data_driven/logistic_regression/`.

| Section | Setting | Notes |
|---|---|---|
| Training data | deposits, study-area mask, hold-out %, seed, optional absence points (ROC) | as WofE |
| Evidence layers | tick layers; per layer: **Continuous** (value, optional log10) or **Categorical** (one 0/1 dummy per class, largest class = reference) | ordered classes (e.g. distance classes) can be used as continuous, like ArcSDM's "ordered" evidence |
| Training samples | **All study-area cells** (weighted LR) or **Deposits + random non-deposit cells**: samples per deposit, minimum distance from deposits, repeats, cross-validation, prior correction | see below |
| Model options | estimator (Firth / ML / ridge λ), variable selection (none / backward AIC / backward LR test with p-to-remove), missing evidence (exclude / area-weighted mean), unit cell, outputs, prefix | |

*Weighted LR* is ordinary LR on every study-area cell; with categorical evidence only, identical cells are pooled into
unique conditions weighted by their area (same likelihood, much faster — ArcSDM `sdmlr.exe`). *Sampled* LR fits
training deposits against random non-deposit cells ≥ the minimum distance from every known deposit (Carranza & Hale
2001; Carranza et al. 2008; ArcSDM *Create Random Points*); case-control sampling shifts only the intercept, which the
**prior correction** β0* = β0 − ln[(1 − τ)/τ · ȳ/(1 − ȳ)] removes (King & Zeng 2001). Several random draws (Zuo & Wang
2020) are pooled with Rubin's rules (the spread between draws is added to the uncertainty).

Estimation: Newton–Raphson; **Firth** bias reduction (Firth 1993; Heinze & Schemper 2002) gives finite estimates when
a class has no deposits (separation) and with few deposits — the default. Inference: Wald z, odds ratios with 95 % CI,
penalised likelihood-ratio test per layer, GVIF collinearity (Fox & Monette 1992), Hosmer–Lemeshow test, McFadden R²,
AIC/BIC, predicted vs observed deposits, K-S check of unique conditions (Agterberg et al. 1993), events-per-variable
warning (< 10 deposits per coefficient; Peduzzi et al. 1996). Sampled design: ArcSDM-style report (confusion matrix,
training AUC, leave-one-out — or 10-fold above 300 samples — cross-validation).

Outputs: `<prefix>_post` (probability per unit cell, P_unit = 1 − (1 − P_cell)^(A_unit/A_cell)), `_post_std`
(delta method p(1 − p)·√(xᵀ Cov x)), `_tvalue` (P / s(P), as ArcSDM's LRTValue), optional `_logit`; tables
coefficients, model_fit, variables, hosmer_lemeshow, correlation (+ selection, repeats, sample_validation).

**Verification against ArcSDM**: the maximum-likelihood engine reproduces `sdmlr.exe` (ArcSDM 5.03 classic Logistic
Regression) on its own unique-conditions case data — coefficients ±5·10⁻⁶, standard deviations, probabilities and
s(P) to all 5 printed decimals (`tests/test_logistic.py::test_ml_reproduces_arcsdm_sdmlr_exe`). Differences from the
ArcSDM tools: sdmlr counts deposits as successes and *area* as failures (trials = area + points; we use cells, the
exact binomial), stops at |score| < 0.01 and takes the covariance from the previous iteration; ArcSDM's classic tool
regresses WofE-generalised classes only (here also continuous values and Firth); in ArcGIS Pro its missing-data
substitution is inactive (string/number comparison) and areas are divided by the unit cell twice when it is not 1 km².
ArcSDM's experimental *Logistic Regression Train* (scikit-learn, class_weight "balanced", output = decision function)
gives uncalibrated scores; the prior correction here gives probabilities.

References: Agterberg (1974) Math. Geol. 6; Agterberg, Bonham-Carter, Cheng & Wright (1993) in Davis & Herzfeld (eds)
*Computers in Geology — 25 Years of Progress*, 13–32; Bonham-Carter (1994) *GIS for Geoscientists*; Carranza & Hale
(2001) Explor. Min. Geol. 10; Carranza, Hale & Faassen (2008) Ore Geol. Rev. 33; Firth (1993) Biometrika 80; Heinze &
Schemper (2002) Stat. Med. 21; King & Zeng (2001) Polit. Anal. 9; Hosmer, Lemeshow & Sturdivant (2013) *Applied
Logistic Regression*; Fox & Monette (1992) JASA 87; Peduzzi et al. (1996) J. Clin. Epidemiol. 49; Rubin (1987);
Zuo & Wang (2020) Nat. Resour. Res. 29; ArcSDM (github.com/gtkfi/ArcSDM; Kemp et al. 2001; Sawatzky et al. 2004).

## 4c. Random Forest (Data-driven ▸ Machine learning ▸ Random Forest…)

Breiman (2001): an ensemble of classification trees, each grown on a bootstrap sample of the training data with a
random subset of evidence columns tried at every split. Training data = training-deposit cells (class 1) + random
non-deposit cells (class 0) away from known deposits (Carranza & Laborte 2015; Carranza et al. 2008). Code:
`python/logic/data_driven/random_forest/` (scikit-learn `RandomForestClassifier`); evidence encoding shared by the
machine-learning methods: `python/logic/common/features.py` (continuous = values, categorical = one 0/1 column per
class; missing evidence excluded or set to the median / most frequent class).

| Section | Setting (default) |
|---|---|
| Training data | deposits, mask, hold-out (20 %, spatially stratified, seed), thinning, absence points (ROC) — as WofE / LR |
| Evidence layers | tick layers; *Continuous* or *Categorical (one-hot)* |
| Training samples | non-deposit samples per deposit (1), min. distance from deposits (4 cells), min. spacing between samples (auto), random samples / repeats (5 — one forest per draw, all trees pooled; Zuo & Wang 2020), cross-validation (none = out-of-bag; 10-fold, 3-fold, leave-one-out ≤ 150 samples) |
| Model options | number of trees (500), features per split (√p; log2 p, p/3, all = bagging), min. samples per leaf (1), max. depth (unlimited), class weight (none / balanced), missing evidence, outputs, prefix (`rf`) |

Outputs: `<prefix>_score` = mean tree vote (share of trees voting "deposit", 0–1 — a prospectivity **ranking**
that depends on the deposit / non-deposit ratio, not a probability per unit cell), `_score_std` = standard
deviation of the tree votes (model uncertainty), training / test points. Results dialog: training samples, **out-of-bag
AUC** (each sample predicted only by the trees that did not use it — an internal test set, Breiman 2001),
cross-validated AUC, ROC AUC of the training / test deposits on the map; tabs *Summary*, *Variable importance*
(permutation = drop of the OOB AUC when a layer is shuffled, mean ± s.d., with a bar chart; impurity = share of the
Gini decrease, biased towards layers with many values — Strobl et al. 2007), *Sample validation* (OOB confusion
matrix, accuracy, AUC, CV), *Repeats*; ROC / AUC…; Excel export.

The training-deposit AUC on the map is close to 1 because the forest has seen those deposits — judge the model by the
**test** and **out-of-bag** AUC. Carlin benchmark (30 % spatial hold-out, seed 42): test AUC 0.893 (LR 0.898, WofE 0.824).

References: Breiman (2001) Machine Learning 45; Carranza & Laborte (2015) Comput. Geosci. 74; Rodriguez-Galiano et al.
(2015) Ore Geol. Rev. 71; McKay & Harris (2016) Nat. Resour. Res. 25; Strobl et al. (2007) BMC Bioinformatics 8;
Zuo & Wang (2020) Nat. Resour. Res. 29.

## 4d. Support Vector Machine (Data-driven ▸ Machine learning ▸ Support Vector Machine…)

Vapnik (1995), Cortes & Vapnik (1995): the soft-margin SVM finds the boundary f(x) = Σ αᵢyᵢK(xᵢ, x) + b with the
largest margin between deposit and non-deposit samples in the feature space of a kernel K; C penalises training
errors. MPM use as Zuo & Carranza (2011), Abedi et al. (2012), Rodriguez-Galiano et al. (2015). Code:
`python/logic/data_driven/svm/` (scikit-learn `SVC` = LIBSVM, Chang & Lin 2011); same training samples and evidence
encoding as the random forest. ArcSDM's SVM tool (RBF, C = 1, γ = 1/p, z-scores, leave-one-out accuracy, decision
value as output) is the special case *tuning off, C = 1, γ = auto, z-score*.

| Section | Setting (default) |
|---|---|
| Training data | deposits, mask, hold-out (20 %, spatially stratified, seed), thinning, absence points (ROC) — as WofE / LR / RF |
| Evidence layers | tick layers; *Continuous* or *Categorical (one-hot)* — use log-transformed geochemistry |
| Training samples | non-deposit samples per deposit (1), min. distance from deposits (4 cells), min. spacing (auto), random samples / repeats (5 — one SVM per draw, scores averaged), cross-validation (5-fold; 10-fold, 3-fold, leave-one-out ≤ 300 samples) |
| Model options | kernel (RBF; linear, polynomial, sigmoid), **tune C and γ** (on: 7 × 6 grid C = 2⁻³ … 2⁹, γ = γ0·2⁻⁶ … γ0·2⁴, 5-fold cross-validated AUC on the first sample; Hsu et al. 2003), C (1), γ (auto γ0 = 1 / (p · mean column variance)), degree / coef0 (polynomial, sigmoid), class weight (none / balanced), scaling (z-score with study-area statistics; min–max; none), missing evidence, outputs, prefix (`svm`) |

Outputs: `<prefix>_score` = Platt-calibrated decision value P = 1 / (1 + exp(A·f + B)) averaged over the samples
(0–1 **ranking**, depends on the sample ratio; the sigmoid is fitted to the held-out cross-validation decision values
with the algorithm of Lin, Lin & Weng 2007 and is always increasing in f, so the score ranks cells exactly as f(x)),
`_score_std` = s.d. of the score over the random samples (repeats ≥ 2), optional `_decision` = mean f(x) (ArcSDM's
output; > 0 = deposit side), training / test points. Results dialog: samples, **cross-validated AUC**, C / γ, ROC AUC
of the training / test deposits; tabs *Summary*, *Parameter tuning* (table + CV-AUC heat map over log₂C × log₂γ),
*Variable importance* (drop of the cross-validated AUC when a layer is shuffled in the held-out folds; linear kernel:
also the weights of the scaled columns), *Sample validation* (training and CV confusion at f = 0, accuracy, AUC,
support vectors, bounded support vectors, Platt A / B), *Repeats*; ROC / AUC…; Excel export.

Carlin benchmark (30 % spatial hold-out, seed 42, 1:5 samples, 3 repeats, tuned): training AUC 0.904, test AUC 0.901.

References: Vapnik (1995) The Nature of Statistical Learning Theory; Cortes & Vapnik (1995) Machine Learning 20;
Platt (1999) in Advances in Large Margin Classifiers; Lin, Lin & Weng (2007) Machine Learning 68; Hsu, Chang & Lin
(2003) A practical guide to support vector classification; Chang & Lin (2011) ACM TIST 2; Zuo & Carranza (2011)
Comput. Geosci. 37; Abedi et al. (2012) Comput. Geosci. 46; Rodriguez-Galiano et al. (2015) Ore Geol. Rev. 71.

## 4e. XGBoost (Data-driven ▸ Machine learning ▸ XGBoost…)

Extreme gradient boosting (Friedman 2001; Chen & Guestrin 2016): an additive model of shallow regression trees on the
log-odds scale, each tree fitted to the gradient and Hessian of the logistic loss of the trees before it, with
shrinkage η, row / column subsampling and L1 / L2 / split-loss regularisation; missing values get a learned default
direction. MPM use: Parsa (2021). Code: `python/logic/data_driven/xgboost_mpm/` (xgboost 3.2, package name avoids
shadowing the library); same training samples and evidence encoding as RF / SVM.

| Section | Setting (default) |
|---|---|
| Training data | deposits, mask, hold-out (20 %, spatially stratified, seed), thinning, absence points (ROC) |
| Evidence layers | tick layers; *Continuous* or *Categorical (one-hot)*; **monotone constraint** per continuous layer (none / increasing / decreasing — e.g. decreasing with distance to faults) |
| Training samples | non-deposit samples per deposit (1), min. distance from deposits (4 cells), min. spacing (auto), repeats (5 — one booster per draw, scores averaged), cross-validation (5-fold; 10, 3) |
| Model options | max. number of trees (500) with **early stopping** (50 rounds) on the mean held-out AUC of the folds, learning rate η (0.05), max. depth (3), min. child weight (1), min. split loss γ (0), row / column subsample (0.8 / 0.8), L2 λ (1), L1 α (0), class weight (none / balanced = scale_pos_weight), missing evidence (exclude / impute / **keep as missing**), **device (auto = NVIDIA GPU via CUDA if one is detected, else CPU; CPU; GPU)**, outputs, prefix (`xgb`) |

Number of trees: all folds are boosted in lockstep; K* = best round of the mean held-out AUC curve (stop when it has
not improved for the patience). The final booster uses K* trees on all samples. Outputs: `<prefix>_score` = predicted
probability averaged over the samples (0–1 **ranking**, depends on the sample ratio), `_score_std` (repeats ≥ 2),
optional `_shap_<layer>` = TreeSHAP contribution of every evidence layer to the log-odds (Lundberg et al. 2020; the
contributions plus a constant add up to the logit of the score), training / test points. Results dialog: samples,
cross-validated AUC, number of trees, device, ROC AUC of training / test deposits; tabs *Summary*, *Boosting curve*
(training and held-out AUC per tree), *Variable importance* (mean |SHAP| share, effect direction = rank correlation of
value and SHAP, gain share, permutation drop of the CV AUC), *Sample validation*, *Repeats*; ROC / AUC…; Excel export.

GPU: the xgboost wheel contains CUDA; on a PC with an NVIDIA GPU and driver, *auto* trains and predicts on the GPU
(probed once per session; the dialog shows the detected device). On a PC without one it runs on the CPU — no setting
needs to be changed. Carlin benchmark (30 % spatial hold-out, seed 42, 1:5 samples, 3 repeats, distance to faults
decreasing): training AUC 0.947, test AUC 0.893.

References: Friedman (2001) Ann. Stat. 29; Chen & Guestrin (2016) Proc. 22nd ACM SIGKDD; Lundberg & Lee (2017) NIPS 30;
Lundberg et al. (2020) Nat. Mach. Intell. 2; Parsa (2021) J. Geochem. Explor. 228.

## 4f. Relevance Vector Machine (Data-driven ▸ Machine learning ▸ Relevance Vector Machine…)

Tipping (2001): sparse Bayesian learning. The deposit probability is σ(Σ w_m φ_m(x)) on basis functions φ = [1,
K(x, x_1) … K(x, x_N)] (Gaussian RBF kernel centred on every training sample) or φ = [1, x] (linear basis = sparse
Bayesian logistic regression with automatic relevance determination). Every weight has its own Gaussian prior
precision α_m; maximising the Bayesian evidence drives most α_m to ∞ and removes their basis functions — the few
remaining training samples are the **relevance vectors**. Code: `python/logic/data_driven/rvm/` (own NumPy
implementation): Laplace approximation (IRLS) of the posterior, the fast sequential marginal-likelihood algorithm of
Tipping & Faul (2003) (add / re-estimate / delete the basis function with the largest evidence gain; verified against
the original re-estimation α ← γ/μ²), moderated prediction P = σ(a / √(1 + πs²/8)) (MacKay 1992).

| Section | Setting (default) |
|---|---|
| Training data | deposits, mask, hold-out (20 %, spatially stratified, seed), thinning, absence points (ROC) |
| Evidence layers | *Continuous* or *Categorical (one-hot)*; z-scores with study-area statistics |
| Training samples | non-deposit samples per deposit (1), min. distance from deposits (4 cells), min. spacing (auto), repeats (5), cross-validation (5-fold; 10, 3) |
| Model options | basis functions (RBF on the training samples; linear = relevance determination of the evidence columns), tune the kernel width γ (γ0·2⁻⁶ … γ0·2⁴ by 5-fold CV AUC), γ (auto = 1/p), missing evidence, outputs, prefix (`rvm`) |

Outputs: `<prefix>_score` (moderated probability averaged over the samples — a ranking that depends on the sample
ratio), `_pred_sd` (Bayesian predictive s.d. of the probability: large where the evidence is unlike the relevance
vectors), `_score_std` (s.d. over the samples), `relevance_vectors` point layer (yellow diamonds, with weights and
evidence values) and training / test points. Results dialog: training samples, relevance vectors / samples,
cross-validated AUC, ROC AUC of training / test deposits; tabs *Summary*, *Relevance vectors*, *Kernel width*,
*Variable importance* (CV permutation; linear basis: retained columns and weights), *Sample validation*, *Repeats*.

Carlin benchmark (30 % spatial hold-out, seed 42, 1:3 samples, 3 repeats): training AUC 0.909, test AUC 0.898;
4 relevance vectors of 75 samples in the GUI test (one deposit prototype in favourable host with high Sb, three barren
prototypes outside the host with low Sb).

References: Tipping (2001) J. Mach. Learn. Res. 1; Tipping & Faul (2003) Proc. AISTATS; MacKay (1992) Neural Comput.
4; Bishop (2006) Pattern Recognition and Machine Learning.

## 4g. Neural Network (MLP) (Data-driven ▸ Machine learning ▸ Neural Network (MLP)…)

Multilayer perceptron (Rumelhart et al. 1986): fully connected hidden layers h = g(Wh + b) and a sigmoid output,
trained by back-propagation with Adam (Kingma & Ba 2015) on the binary cross-entropy + L2 penalty. Architecture and
defaults follow eis_toolkit's `train_MLP_classifier` (dense layers, ReLU, Adam 0.001, batch 32, early stopping),
implemented with scikit-learn `MLPClassifier` (no TensorFlow; user decision). Code: `python/logic/data_driven/mlp/`.
MPM use: Brown et al. (2000), Porwal et al. (2003). Two changes for MPM data: (1) the number of epochs is chosen by the
**mean held-out AUC** of k folds trained epoch by epoch in lockstep (scikit-learn's own early stopping uses accuracy,
uninformative with imbalanced classes; eis_toolkit's Keras `validation_split` takes the last rows unshuffled); (2) an
**ensemble** of networks with different initial weights per random sample (small networks are unstable).

| Section | Setting (default) |
|---|---|
| Training data / evidence / training samples | as RF / SVM; z-scores; repeats (5), cross-validation (5-fold; 10, 3) |
| Model options | hidden layers (`32, 16`), activation (ReLU; tanh, sigmoid), learning rate (0.001), L2 penalty α (0.0001), batch size (32), max. epochs (500), early stopping (20 epochs), networks per sample (5), missing evidence, outputs, prefix (`mlp`) |

Outputs: `<prefix>_score` (mean output of all networks), `_score_std` (s.d. of the network outputs = model
uncertainty), training / test points. Results dialog: samples, CV AUC, epochs, networks, ROC AUC training / test; tabs
*Summary*, *Learning curve*, *Variable importance* (CV permutation), *Sample validation*, *Repeats*. Carlin GUI test
(25 training deposits, 1:3, 2 × 3 networks): epochs 61, training AUC 0.914, test AUC 0.898.

References: Rumelhart et al. (1986) Nature 323; Kingma & Ba (2015) ICLR; Brown et al. (2000) Aust. J. Earth Sci. 47;
Porwal et al. (2003) Nat. Resour. Res. 12; eis_toolkit (EIS Horizon project, prediction/mlp.py).

## 4h. Extreme Learning Machine (Data-driven ▸ Machine learning ▸ Extreme Learning Machine…)

Huang et al. (2006, 2012): a single-hidden-layer network whose input weights and biases are drawn at random and never
trained; only the output weights are computed, in closed form, by regularised least squares β = (HᵀH + I/C)⁻¹Hᵀt
(H = hidden-layer outputs of the training samples, t ∈ {0, 1}). Training is one linear solve. Code:
`python/logic/data_driven/elm/` (own NumPy code). MPM use: Chen & Wu (2017). Because one ELM depends on its random
hidden layer, an **ensemble** of ELMs is averaged; the ensemble output is mapped to [0, 1] by a Platt sigmoid fitted to
the cross-validated outputs (monotone; same routine as the SVM). C is tuned by 5-fold CV AUC over 2⁻⁶ … 2¹².

| Section | Setting (default) |
|---|---|
| Training data / evidence / training samples | as RF / SVM; z-scores; repeats (5), cross-validation (5-fold; 10, 3) |
| Model options | hidden neurons (100), activation (sigmoid; tanh, ReLU, Gaussian RBF with random centres), tune C (on) / C (1), ELMs per sample (10), missing evidence, outputs, prefix (`elm`) |

Outputs: `<prefix>_score` (calibrated ensemble output, mean over the samples), `_score_std` (s.d. of the ensemble
members), training / test points. Results dialog tabs: *Summary*, *Regularisation* (CV AUC over log₂C),
*Variable importance*, *Sample validation*, *Repeats*. Carlin GUI test (1:3, 2 × 5 ELMs): C = 0.25, training AUC
0.903, test AUC 0.912.

References: Huang, Zhu & Siew (2006) Neurocomputing 70; Huang et al. (2012) IEEE Trans. Syst. Man Cybern. B 42;
Chen & Wu (2017) Ore Geol. Rev. 80.

## 4i. Convolutional Neural Network (Data-driven ▸ Machine learning ▸ Convolutional Neural Network (CNN)…)

Patch-based CNN (LeCun et al. 1998; MPM: Sun et al. 2020, Li et al. 2021) in **TensorFlow / Keras** (user decision;
already installed for eis_toolkit; TensorFlow 2.15.1 / Keras 2.15.0 are in the offline wheels and **bundled in the
release build**; the tool still shows a clear message if TensorFlow is missing). Every cell is represented by the k × k window of all evidence layers around it (channels =
z-scored columns, one-hot classes), so the network learns the spatial context (shape and neighbourhood of anomalies,
contacts, structures). Architecture: Conv2D(f₁, 3×3) → Conv2D(f₂, 3×3) → MaxPool → Dense → Dropout → sigmoid; Adam;
training patches augmented by the 8 rotations / reflections (D4). Epochs: k-fold CV with EarlyStopping on the
held-out AUC, E* = median best epoch; ensemble of networks. Code: `python/logic/data_driven/cnn/`. TensorFlow uses a
visible NVIDIA GPU automatically; TensorFlow ≥ 2.11 has no GPU support on native Windows (CPU there).

| Section | Setting (default) |
|---|---|
| Training data / evidence / samples | as the other ML methods; repeats (5), cross-validation (3-fold; 5, 10) |
| Model options | patch size (9 cells; window shown in map units), convolution filters (`8, 16`), dense neurons (16), dropout (0.5), learning rate (0.0003), batch (32), max. epochs (200), early stopping (20), augmentation (on), networks per sample (2), missing evidence, outputs, prefix (`cnn`) |

The small default network was chosen on Carlin (30 % spatial hold-out): 16/32 filters, 32 neurons, learning rate
0.001 over-fitted within 2–5 epochs (test AUC 0.863); 8/16, 16, dropout 0.5, 0.0003 gave 0.888 with a quarter of the
weights. Results dialog tabs: *Summary*, *Learning curve*, *Variable importance*, *Sample validation*, *Repeats*.

## 4j. Bayesian Network (Data-driven ▸ Statistical ▸ Bayesian Network…)

Bayesian network classifiers (Porwal, Carranza & Hale 2006), own code (NumPy + networkx; pgmpy is unusable without
PyTorch). **Naive Bayes**: P(D | x) ∝ P(D) Π P(xᵢ | D) — trained on all study-area cells it is exactly weights of
evidence (test). **TAN** (Friedman et al. 1997): each evidence layer may also depend on one other layer; the tree
maximises the conditional mutual information I(Xᵢ; Xⱼ | D) (Chow & Liu 1968 maximum spanning tree), so correlated
layers are not double-counted. Continuous layers → equal-area (quantile) classes; categorical layers keep their
classes; missing evidence = own class or excluded. Tables with a Dirichlet prior (equivalent sample size α);
posterior s.d. from Dirichlet draws of the tables.

| Section | Setting (default) |
|---|---|
| Training data / evidence | deposits, mask, hold-out (20 %), seed, thinning, absence points; continuous / categorical |
| Network and training cells | structure (TAN; naive Bayes), training cells (all study-area cells; deposits + random non-deposit cells with ratio / min. distance), classes of continuous layers (5), Dirichlet α (1), uncertainty draws (50), unit cell (1 km²), missing evidence (own class), outputs, prefix (`bn`) |

Outputs: `post` (posterior per raster cell), `post_unit` (per unit cell), `post_sd`, training / test points.
Results: *Summary*, *Network* (drawing: D → every layer, TAN arrows with width ∝ conditional MI), *Probability tables*,
*Evidence layers* (mutual information, parents, AUC drop without the layer), *Structure comparison* (NB vs TAN
log-likelihood, parameters, BIC, AUC). Carlin GUI test: TAN edges As → Sb (I = 0.32), host → distance to faults,
host → As; observed / predicted deposits 0.95 (WofE 0.59), test AUC 0.877.

References: Porwal et al. (2006) Comput. Geosci. 32; Friedman, Geiger & Goldszmidt (1997) Mach. Learn. 29; Chow & Liu
(1968) IEEE Trans. Inf. Theory 14; Heckerman, Geiger & Chickering (1995) Mach. Learn. 20.

## 4k. Knowledge-driven overlays (Knowledge-driven ▸ Overlay / Fuzzy)

Deposits are **not** used to build these maps — only (optionally) to validate them (ROC / success rate / P–A of all
deposits). Code: `python/logic/knowledge_driven/<method>/`.

**Fuzzy Logic Overlay** (Bonham-Carter 1994; An et al. 1991): per evidence layer a membership function — linear (a →
b; a > b = decreasing), S / Z-shaped, logistic (inflection, slope), Gaussian (centre, σ), ArcGIS large / small
(midpoint, spread), or a membership per class (`class=membership; …`); defaults from the 2nd–98th percentiles of the
data (to be edited). scikit-fuzzy (`smf`, `zmf`, `sigmf`, `gaussmf`) provides the standard curves. Layers with the same
**group** name (mineral-system component: source, pathway, trap, footprint) are combined by the within-group operator,
the groups by the between-group operator: fuzzy AND (min), OR (max), algebraic product, algebraic sum, γ
(sum^γ·product^(1−γ)), geometric average. The γ and sum operators equal eis_toolkit's (test). Missing evidence:
exclude or membership 0.5. Outputs: favourability, group maps, optional membership maps; results dialog with the
inference network, membership-function plots over the data histograms and membership statistics.

**Boolean Logic Overlay** (Bonham-Carter 1994): per layer a condition (≥, ≤, between, outside, in classes; defaults =
quartiles), AND / OR within and between groups. Outputs: favourable area (1/0), number of satisfied conditions (a
ranking for ROC), optional condition maps; table of area %, deposits inside and deposit-share / area-share ratio for
every condition and the combined map.

**Logistic Fuzzification + Geometric Average** (Yousefi & Carranza 2015a, b): continuous layers fuzzified by
F = 1/(1 + e^(−s(x − i))), i = (min + max)/2, s = ±9.2/(max − min) (range → 0.01 … 0.99; positive or negative
evidence), categorical layers by class scores; combination by the (weighted) geometric average; weights equal, custom
or from the P–A plot of each layer (w = ln(Pr/Oa), Yousefi & Carranza 2015c; layers with Nd ≤ 1 get 0).

Demo results (all deposits; `docs/python/tools/*.docx`): Carlin — fuzzy AUC 0.778, Boolean target 15.7 % of the area
with 66 % of the deposits, logistic + geometric average AUC 0.850 (P–A weights 0.780); Curnamona — fuzzy 0.909,
Boolean 21.8 % / 71 %, logistic + geometric average 0.893.

### Offline coordinate systems

All coordinate systems come from the EPSG databases (`proj.db`) bundled with rasterio, fiona and pyproj; nothing is
downloaded (`PROJ_NETWORK=OFF`). Other GIS software (PostgreSQL/PostGIS, QGIS, ArcGIS, GRASS, OSGeo4W, Anaconda) often
sets `PROJ_LIB` / `PROJ_DATA` / `GDAL_DATA` for the whole PC; GDAL would then use that foreign, often incompatible
database, read the demo GeoTIFFs as a datum-less `LOCAL_CS["GDA94 / MGA zone 54"]` and fail with "Cannot find
coordinate operations". `python/utils/geo_env.py` therefore removes such variables at start-up (logged; set
`MPM_KEEP_PROJ_ENV=1` to keep them), checks that EPSG codes resolve (error message in the log otherwise), and a raster
or vector whose coordinate system is **unrecognised** — missing, unparsable, datum-less (engineering / LOCAL_CS) or
GDAL's "Undefined … SRS" placeholder — or cannot be transformed to the project CRS is loaded as stored and treated as
being in the current project CRS (warning in the log). This applies when opening a project and to *Add Data*.

## 5. Validation (Validation ▸ ROC / AUC validation…)

Works for any raster (higher = more prospective) and deposit layer(s). Dialog tabs: **ROC curve** with AUC (training
and test), **success-rate** curve, **prediction–area (P-A) plot** with the intersection point, **summary** table;
export as PNG/SVG/PDF and CSV. Code: `python/evaluation/metrics.py` (cell-based: deposit cells vs all other study-area cells,
or vs optional **absence points** as in ArcSDM's ROC tool). The ROC AUC (Mann–Whitney, ties ½) comes with the DeLong
et al. (1988) standard error and 95 % CI. ArcSDM's ROC tool instead draws as many random background points as deposits
10,000 times and reports the AUC of the median curve with a percentile interval (and orders tied scores arbitrarily);
using every background cell is the deterministic limit of that procedure.

### Target delineation (Validation ▸ Target delineation (P-A / C-A)…)

Turns **any raster** — the output of any method or an evidence layer — into a binary target map:
**1 = value ≥ threshold (target, shown red), 0 = below (transparent)**, nodata outside the study area.

| Method | Threshold | Needs |
|---|---|---|
| **P-A intersection** (Yousefi & Carranza 2015) | where the prediction rate (% deposits with value ≥ t) meets 100 − occupied area (% area with value ≥ t) | deposit layer |
| **C-A fractal** (Cheng et al. 1994) | breakpoint of log10 A(≥ v) vs log10 v, fitted with 2–5 straight segments (exact least-squares segmentation); default = break before the highest-valued segment, any breakpoint can be chosen | nothing (deposits optional, only for statistics); rasters with values ≤ 0 are shifted before log10 |

The graph shows the automatic threshold as a **vertical dashed line**; with the cross cursor, click or drag on the graph to
move it, or type the value (*Automatic* resets it). Live statistics: target area (% and km²) and deposits inside the
target. **Confirm** writes `results\target_<pa|ca>_<MMDD>_<HHMMSS>\` with `target.tif` (uint8, class colours in
`target.classes.csv`), `plot.png` and `target_report.csv`, and adds the layer to the map.
Code: `python/evaluation/targets.py` (logic, tested in `tests/test_targets.py`) and `python/gui/dialogs/target_dialog.py`.

## 6. Settings (Settings ▸ General settings…, Ctrl+,)

Pages on the left, each page has its own **Apply** button (enabled after a change). Stored per Windows user
(registry `HKCU\Software\MPM\MPM Studio`, keys `ui/language`, `ui/theme`).

| Page | Choices | Takes effect |
|---|---|---|
| **Language** | English, 简体中文 (every `python/i18n/<code>.json` appears automatically) | after a restart — Apply offers *Restart now* (asks to save, reopens the current project) |
| **Theme** | Dark, Light | immediately (window, icons, map legend; charts opened afterwards). The map background has the theme's window colour while the map has no layers (start-up) and is white as soon as there is a layer, in both themes |

### Translations (adding a language)

All user-visible texts are written in English in the code and wrapped in `tr()`; the English text is the key:

```python
from i18n import tr, N_
QLabel(tr("Cell size"))
log.info(tr("Opened project {path} ({n} layers)", path=p, n=n))   # named placeholders, never f-strings inside tr()
CATEGORIES = [N_("Data-driven"), ...]                             # constants: mark with N_, show with tr(value)
```

Each other language is one UTF-8 JSON file in `python/i18n/` (`zh_CN.json`):

```json
{"_meta": {"name": "简体中文", "english": "Chinese (Simplified)", "qt": "zh_CN"},
 "Cell size": "像元大小",
 "Opened project {path} ({n} layers)": "已打开项目 {path}（{n} 个图层）"}
```

* Empty or missing entries are shown in English, so a catalog can be translated step by step.
* `_meta.qt` selects Qt's own catalog for standard buttons and file dialogs (`qtbase_<qt>.qm` / `qt_<qt>.qm` from PyQt5).
* New language: `..\.venv\Scripts\python.exe tools\i18n_update.py --new ko` (creates `i18n/ko.json` with every
  text), set `_meta`, translate the values, restart and choose it in Settings ▸ General settings ▸ Language.
* After changing GUI texts run `tools\i18n_update.py` (adds new keys to all catalogs and prints
  translated / new / unused counts; `--missing zh_CN` lists the untranslated texts, `--prune` drops unused ones).
* The release build bundles `i18n/*.json` (PyInstaller spec), so a catalog can also be dropped into
  `MPM Studio\_internal\i18n\` of an installed copy.
* Charts use DejaVu Sans with Microsoft YaHei as fallback for Chinese characters.

## 7. Code structure and how to add a method

```
python/
├─ main.py, version.py        entry point (--version, --selftest), application name/version
├─ run_mpm.bat                start from source · install_env.bat: offline .venv setup · build_release.bat: release
├─ requirements*.txt          direct / locked / offline package lists
├─ whl/                       offline wheel set (+ runtime/: Python and VC++ installers)
├─ core/                      application framework (no algorithms)
│  ├─ layers.py               RasterLayer, VectorLayer, LayerStore (signals), project files
│  ├─ symbology.py            renderers without Qt: single / unique / stretched / categorized, classification breaks
│  ├─ methods.py              MPMMethod base class, @register, REGISTRY, CATEGORIES, PLANNED, plug-in discovery
│  └─ worker.py               Task (QThread) for long computations
├─ gui/                       generic user interface
│  ├─ main_window.py          menus/toolbar/docks; method menus are generated from the registry
│  ├─ map_canvas.py           map view + coordinate bar
│  ├─ toc.py, widgets.py
│  ├─ theme.py, app_settings.py dark/light palettes + icons, stored user preferences
│  ├─ dialogs/                layer, data tools, identify, measure, evaluation (ROC/AUC), target, settings dialogs
│  └─ resources/              icon, stylesheet images
├─ logic/                     ★ all MPM methods
│  ├─ data_driven/            one folder per method
│  │  ├─ wofe/                model.py (algorithm) · plugin.py (registration) · dialogs.py (UI)
│  │  ├─ logistic_regression/ same layout (weighted / sampled LR, Firth, tests)
│  │  ├─ random_forest/       same layout (scikit-learn forest, tree-vote std, OOB, importances)
│  │  ├─ svm/                 same layout (scikit-learn SVC, C/γ tuning, Platt score, CV importance)
│  │  ├─ xgboost_mpm/         same layout (xgboost, early stopping, monotone constraints, SHAP, GPU auto)
│  │  ├─ rvm/                 same layout (sparse Bayesian learning, Tipping & Faul fast algorithm, relevance vectors)
│  │  ├─ mlp/                 same layout (scikit-learn MLP, AUC early stopping, network ensemble)
│  │  ├─ elm/                 same layout (random hidden layer, ridge output weights, ensemble, Platt score)
│  │  ├─ cnn/                 same layout (TensorFlow patch CNN, D4 augmentation, CV early stopping)
│  │  └─ bayesian_network/    same layout (naive Bayes / TAN, Dirichlet tables and uncertainty)
│  ├─ knowledge_driven/       fuzzy_logic/, index_overlay/, ahp/, topsis/ … (planned)
│  ├─ hybrid/                 neuro_fuzzy/, fuzzy_wofe/ … (planned)
│  ├─ common/                 GUI-free helpers shared by methods (training.py: train/test split, non-deposit
│  │                          sampling, training/test point layers; features.py: evidence → feature matrix)
│  └─ method_template/        copy-me skeleton of a method (runnable example, not loaded)
├─ i18n/                      tr() / N_() + one JSON catalog per language (zh_CN.json)
├─ evaluation/metrics.py      ROC/AUC, success-rate, P-A (shared by all methods and Validation menu)
├─ utils/                     data preparation: convert, reclass, raster_tools, gdal_rat, prepare_demo_data
├─ tools/                     developer scripts: make_icon, pip_check, smoke_test, i18n_update, benchmark_methods
├─ packaging/                 PyInstaller spec, runtime hook, build_release.py
├─ tests/                     pytest
└─ distribute/               generated by build_release.bat: build/ (temporary), dist/ (application), release/ (zip)
```

Dependency rule: `logic/*/model.py` and `logic/common` use only numpy/scipy/scikit-learn (+ `utils`, `evaluation`) —
never Qt, so algorithms stay testable and reusable. `plugin.py` connects a model to the app; `dialogs.py` holds its UI.

**Adding a method** (example: random forest):

1. Copy `python/logic/method_template/` to `python/logic/data_driven/random_forest/`
   (category folder = `data_driven`, `knowledge_driven` or `hybrid`).
2. `model.py`: the algorithm + a test in `python/tests/`.
3. `plugin.py`: set `key`, `name`, `category` ("Data-driven" | "Knowledge-driven" | "Hybrid"), `group` (sub-menu), `icon`;
   build the outputs in `run()` (worker thread — no Qt) and return a `MethodResult` with an `evaluation`
   (`evaluation.metrics.evaluate`) to get the shared ROC/AUC dialog. Data-driven methods split the deposits with
   `logic.common.training.split_deposits` and add `vectors=deposit_point_outputs(...)` to publish the
   training / test points (`OutputVector` → `results/<run_id>/<name>.gpkg`).
4. `dialogs.py`: parameter dialog whose `params()` returns plain data.
5. Remove its line from `core.methods.PLANNED`.

That is all: `core.methods.load_builtin_methods()` finds `logic/<category>/<method>/plugin.py` automatically, the
method appears in the *Knowledge-driven / Data-driven / Hybrid* menu, and the PyInstaller spec bundles everything under
`logic/` (no hidden-import list to maintain). Shared code used by several methods goes to `logic/common/`.

## 8. Tests

Full testing guide (test levels, options, GUI test rules, benchmark, release checks): [how_to_test.md](how_to_test.md).

```bat
cd python
..\.venv\Scripts\python.exe -m pytest tests -q
```

Tests cover: clip raster by mask, polygon fill colour, recent projects, translation catalogs (complete, same placeholders) and light/dark icon re-colouring, WofE formulas (hand calculation), generalisation, zero-deposit handling, missing-data
variance (vs a literal port of ArcSDM's raster algebra), evaluation metrics (DeLong AUC vs brute force, absence points), agreement with
eis_toolkit for categorical/ascending/descending weights, full Carlin model; logistic regression (ML vs statsmodels and vs ArcSDM
`sdmlr.exe` output, Firth vs direct maximisation, penalised LR test, unique conditions = per-cell fit, single binary layer =
WofE contrast, prior correction, repeats/selection, ArcSDM missing-data rule, Carlin model); random forest (tree-vote
mean/std = scikit-learn, signal vs noise importance, OOB / CV, reproducibility, one-hot encoding, Carlin model); support vector machine (Platt = direct minimisation and monotone, decision map =
scikit-learn, scaling, tuning grid, signal vs noise importance, linear weights, repeats, Carlin model); XGBoost (early stopping, SHAP additivity, monotone constraint, native missing
values, GPU → CPU fallback, reproducibility, Carlin model); RVM (fast algorithm = evidence maximum of the re-estimation
algorithm, sparsity, ARD, moderated prediction, Carlin model); MLP (hidden-layer parsing, epoch training learns and is
reproducible, E* = best held-out epoch, ensemble, importance, Carlin model); ELM (output weights = ridge least
squares in both forms, RBF layer, C tuning, ensemble, Carlin model); CNN (D4 augmentation, exact patch windows and
padding, spatial signal, Carlin model); Bayesian network (naive Bayes = WofE posterior, conditional mutual
information, equal-area classes, TAN finds dependencies and fixes double counting, Carlin model); knowledge-driven
overlays (membership functions, operators = eis_toolkit, inference network, Boolean conditions and statistics,
logistic range → 0.01–0.99, weighted geometric mean, P–A weights, Carlin models); natural breaks (Fisher–Jenks optimality vs mapclassify) and reclassification, offscreen GUI workflow (WofE, LR, RF, SVM, XGBoost, RVM, MLP, ELM, CNN, Bayesian network, fuzzy, Boolean,
logistic-geometric, identify, coordinates, middle-button pan) on both demo
projects (set `MPM_SCREENSHOT_DIR` to save screenshots).

**Method performance benchmark (console).** `python/tools/benchmark_methods.py` runs every implemented method on the
demo projects through the same code path as the dialogs, holds out test deposits (default 30 %, spatially stratified),
and prints for the training and test deposits: deposits / negatives, ROC AUC with the DeLong 95 % CI, success-rate
AUC, P-A intersection and ROC values (TPR at FPR 0.05 / 0.10 / 0.20 / 0.30 / 0.50), then a summary table. All methods
are scored on the same cells (cells with missing evidence excluded) and the same deposits.

```bat
..\.venv\Scripts\python.exe tools\benchmark_methods.py                         :: all methods, both demos
..\.venv\Scripts\python.exe tools\benchmark_methods.py --method wofe --dataset carlin --test-pct 20 --split random
..\.venv\Scripts\python.exe tools\benchmark_methods.py --csv bench.csv --roc-dir roc_curves   :: summary + ROC curves as CSV
..\.venv\Scripts\python.exe -m pytest tests\test_benchmark.py -s              :: Carlin benchmark as a test (thresholds)
```

A new method needs one entry in `CONFIGS` (evidence + parameters per demo); otherwise the report says it has no
configuration. The Curnamona demo has only 7 deposits, so its test AUC (2 deposits) is indicative only.
