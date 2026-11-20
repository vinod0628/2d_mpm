# 2D MPM demo datasets

Two open case studies that, together, cover every evidence type and every method family in
[`docs/methods_survey.md`](../docs/methods_survey.md). Each folder keeps the **original files untouched in `raw/`**;
processed, co-registered stacks will go in `processed/` (not created yet).

| | `curnamona_ree/` | `nevada_carlin_au/` |
|---|---|---|
| Target | REE (+U, Cu, Co) | Carlin-type (carbonate-hosted) Au–Ag |
| Region | Northern Curnamona Province, South Australia | North-central Nevada, USA (Great Basin) |
| Source | Zenodo [10.5281/zenodo.17098677](https://doi.org/10.5281/zenodo.17098677) (DEEP-SEAM v1.0, EarthByte; data from Geological Survey of South Australia / SARIG) | USGS Open-File Report 01-221 v1.1, Raines (2001, rev. 2006), <https://pubs.usgs.gov/of/2001/of01-221/> (ArcSDM course data) |
| Licence | GPL-3.0 (Zenodo/GitHub package); underlying data from SA government (SARIG) — check SARIG terms before redistribution | US public domain (USGS) |
| CRS | EPSG:4283 (GDA94 geographic) | Lambert Conformal Conic, Clarke 1866 (NAD27), std parallels 33°/45°, central meridian −117°, false E/N 0 — **no .prj; CRS must be assigned** |
| Grid | 257 × 244 cells, 0.0018522° (~180 m) | mixed: 250 m (geology), 1 km (stream-sed), ~2.4 km / ~4.8 km (geophysics) |
| Extent | 139.27–139.74 E, 30.27–29.82 S (~45 × 50 km) | ~252 × 278 km |
| Deposits | **7** REE occurrences (`GIS/ree.shp`) | **35** Carlin-type Au deposits (`Train2.shp`, has a `RANDOM` field usable for train/test split) |
| Size | 27 MB | 28 MB |

## Evidence layers

### curnamona_ree/raw/Input_Data_Layers
| Group | Files | Notes |
|---|---|---|
| Geology (vector) | `Geology factors/original/`: `Archean_Early_Mesoproterozoic_Faults.shp` (195 lines), `Felsic_granite.shp` (31 polys), `Mesoproterozoic.shp` (57 polys) | → distance-to-fault/granite/contact, categorical units |
| Geochemistry (points) | `Geochemical/ILRRPCA.csv` (615 samples × **27 elements**: Ag As Au BaO Bi Co Cr2O3 Cs Cu La Mo Nb Ni Pb Rb Sb Sc Sn SrO Ta Th U3O8 V W Y Zn Zr), `geochemical_unlabel_all_869_XY.shp` (sample locations) | raw concentrations → clr/ilr, PCA, IDW/kriging |
| Gravity | `SA_GRAV_ONSHORE`, `_1VD`, `_UC1000_RESIDUAL` | continuous rasters |
| Magnetics | `SA_TMI`, `_VRTP`, `_1VD`, `_AGC`, `_AS`, `_BigE`, `_BigT`, `_CAUCHY_3rd`, `_PseudoGrav`, `_TZ`, `_Tilt`, `_Tzz` | 12 derivatives |
| Radiometrics | `SA_RAD_K`, `_TH`, `_U`, `_DOSE` | continuous |
| **Remote sensing** | 15 ASTER geoscience products: AlOH group composition/content, FeOH content, ferric oxide composition/content, ferrous iron index / in MgOH, green vegetation, gypsum, kaolin, MgOH composition/content, opaque, quartz, silica | uint8 (0–254) |
| DEM | `DEM/DEM_9s.tif` | |
| Mask | `GIS/Clip_frame.shp` | study area |
| Authors' outputs | `../Output_Features_Generated/*.csv` | pre-extracted features & PCs from the DEEP-SEAM paper — useful for cross-checking, not required |

### nevada_carlin_au/raw
| Group | Files | Notes |
|---|---|---|
| Geology | `KBGEOL/` (ArcInfo grid, 250 m, 25 lithologic classes; legend in `kbgeoltbl.dbf` incl. a `CARLIN` T/F favourable-host flag, `Kbgeol.avl`), `gbfaults3.shp` (1 674 faults, with azimuth) | categorical + distance to faults |
| Geochemistry (points) | `Naa.shp` — **3 472 NURE stream-sediment samples × 46 elements** (NAA: Ag Al As B Ba … Zn Zr, DNC U) | raw values; 0 = below detection / missing — must be handled |
| Geochemistry (grids) | `SBFACE1/` (continuous, 1 km), `RCLSSB2/` (15 classes, 1 km) | not documented in the files; by name and resolution most likely an interpolated **Sb (antimony)** surface and its reclassification — confirm against OFR 01-221 before use |
| Geophysics | `aeromag/`, `bouguer/` | **pre-classified** grids (5 and 8 classes), not raw values |
| Radiometrics | `potassium/`, `thorium/`, `uranium/` | pre-classified grids (~2.4 km) |
| Expert map | `EXPERT2/` | 3-class knowledge-driven map from the course — handy as a reference for fuzzy/index-overlay results |
| Masks | `studyarea.shp`, `studygrd3/` | |
| Reports | `_Reports/of01-221_v1.1.pdf` (course text with WofE / fuzzy / NN exercises), USGS DDS-11, DDS-41, PP-901, NURE manual | |
| Legacy | `carlin.apr`, `*.avl`, `_Extensions/` (ArcView 3 project/scripts) | not needed |

## Method suitability

| Method family | Curnamona REE | Carlin Au |
|---|---|---|
| Knowledge-driven (Boolean, index overlay, fuzzy, AHP, TOPSIS, DS) | ✅ best (rich continuous layers incl. remote sensing) | ✅ (and an expert map to compare against) |
| WofE / LR / EBF / frequency ratio | ⚠ only 7 deposits → large weight variances; usable for demo | ✅ classic WofE tutorial set (35 deposits) |
| RF / SVM / RVM / MLP / boosting | ⚠ 7 positives → needs PU-learning, one-class, or augmentation | ✅ 35 positives + negative sampling |
| One-class / PU / anomaly detection | ✅ (what the source paper does) | ✅ |
| Remote-sensing evidence | ✅ | ❌ none included |

## Known issues to handle in pre-processing
1. Carlin rasters have no CRS metadata and different resolutions → assign the LCC CRS above, resample to a common grid.
2. Carlin geophysics/radiometrics are already classified (few unique values), so continuous-value fuzzification is limited there.
3. Curnamona rasters are 257×244 and 257×245 (DEM, K) → align to one grid.
4. Curnamona is in geographic degrees → reproject to a metric CRS (e.g. GDA94 / MGA zone 54, EPSG:28354) before distance calculations.
5. NURE `0` values in Carlin geochemistry = below detection / not analysed → treat as censored / missing.
