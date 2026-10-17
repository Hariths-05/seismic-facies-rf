# seismic-facies-rf
3D seismic facies classification pipeline using a Random Forest classifier. Features custom attribute extraction and strict spatial validation to prevent geological data leakage.
```markdown
# 3D Seismic Facies Classification with Physics-Driven Attributes & Spatial-Block Random Forest

End-to-end machine learning pipeline and interactive web application for 3D stratigraphic facies classification on the public **F3 Netherlands North Sea** seismic benchmark (Alaudah et al., 2019). The workflow engineers a 7-channel 3D seismic attribute suite with reflective boundary padding and trains a Random Forest under a **strict inline-block spatial split** to prevent spatial autocorrelation leakage across continuous geological reflectors.

---

## Key Results & Resume Summary

> *"Built a seismic facies classifier on a cropped subset of the public F3 Netherlands volume (labeled benchmark), engineering geophysical attributes (envelope, instantaneous phase/frequency, sweetness, 3D local variance, windowed RMS). Trained a Random Forest with a strict inline-block train/test split (`8.91M` train / `2.78M` test voxels) to prevent spatial leakage, achieving **81.24% strict block accuracy** (**0.81 weighted-F1**, **0.44 macro-F1** across 5 target stratigraphic units); confusion concentrated in **Scruff Group (Class 4)** vs. **Lower North Sea (Class 2)** and **Rijnland/Chalk (Class 3)**, consistent with thin-bed tuning and pinch-out geometry. Feature importance ranked **3D Local Variance (22.94%)** and **Windowed RMS (21.72%)** as dominant."*

---

## Repository Structure

```text
seismic-facies-rf/
├── .streamlit/
│   └── config.toml                        # Minimalist theme configuration for Streamlit UI
├── data/                                  # Ignored by Git (see Data Setup below)
│   ├── raw/
│   │   ├── train/                         # train_seismic.npy, train_labels.npy (401, 701, 255)
│   │   └── test_once/                     # test1_*.npy (200, 701, 255), test2_*.npy (601, 200, 255)
│   └── processed/                         # Cropped volumes, feature stack, and trained RF model
├── figures/
│   ├── 01_eda_slices.png                  # Inline amplitude, stratigraphic labels, and Z-slice
│   ├── 03_attribute_qc.png                # Multi-attribute QC panel on middle inline
│   ├── facies_prediction_result.png       # Unseen Test Inline 175: Seismic vs. Truth vs. RF
│   ├── confusion_matrix.png               # Row-normalized confusion matrix on unseen block
│   └── feature_importance.png             # Ranked Gini feature importances
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb      # Phase 1: Volume inspection, percentiles, class counts
│   ├── 02_cropping_and_saving.ipynb       # Phase 2: Sub-volume cropping & global z-score scaling
│   ├── 03_attribute_extraction.ipynb      # Phase 3: 3D complex trace & structural attributes
│   ├── 04_random_forest_training.ipynb    # Phase 4: Inline-block split & Random Forest training
│   └── 05_visualization.ipynb             # Phase 5: Unseen inline prediction & evaluation plots
├── src/
│   ├── __init__.py
│   ├── config.py                          # Centralized project paths, crop bounds, class metadata
│   └── attributes.py                      # 3D Hilbert, RMS, Variance, and GLCM attribute engine
├── app.py                                 # Interactive Streamlit web application (with Ground Truth)
├── style.css                              # External stylesheet (IBM Plex Sans, flat color-blocking)
├── environment.yml                        # Conda environment specification (Python 3.10)
├── requirements.txt                       # Pip dependencies
└── README.md

```

---

## 1. Data Source & Stratigraphic Classes

* **Benchmark Reference:** Y. Alaudah, P. Michałowicz, M. Alfarraj, and G. AlRegib (2019), *"A Machine-Learning Benchmark for Facies Classification"*, **Interpretation**, 7(3), SE175–SE187.
* **Download Link & License:** Zenodo Record [`3755060`](https://zenodo.org/record/3755060) (Creative Commons Attribution 4.0 International, CC BY 4.0).
* **Survey Geometry:** Offshore Netherlands F3 Block (`Inline × Crossline × Time/Depth Z`), $\Delta\text{IL} = \Delta\text{XL} = 25\text{ m}$, vertical sample interval $\Delta t = 4\text{ ms}$ two-way travel time (TWT).
* **Training Volume (`train_seismic.npy` / `train_labels.npy`):** `(401, 701, 255)` = `71,680,755` voxels, amplitude scaled to `[-1.0, 1.0]` (`1st` percentile `-0.634`, `99th` percentile `0.564`).

| Class ID | Stratigraphic Unit | Full Volume Voxels | Full Volume Share | Role in Training |
| --- | --- | --- | --- | --- |
| `0` | Upper North Sea Group (Shallow / Background) | `20,137,839` | `28.09%` | Masked (`y > 0`) |
| `1` | Middle North Sea Group | `8,519,666` | `11.89%` | Target Class |
| `2` | Lower North Sea Group | `34,831,122` | `48.59%` | Target Class |
| `3` | Rijnland / Chalk Group | `4,760,778` | `6.64%` | Target Class |
| `4` | Scruff Group | `2,350,150` | `3.28%` | Target Class |
| `5` | Zechstein Group (Evaporites / Salt Diapir) | `1,081,200` | `1.51%` | Target Class |

---

## 2. Sub-Volume Cropping, Global Normalization & Spatial Split Strategy

### Sub-Volume Crop Window

To run full 3D attribute extraction and Random Forest training on a laptop CPU while retaining all deep stratigraphic units (Scruff Group pinches out entirely after `IL 234`, and the Zechstein salt diapir is localized within `XL[430:638], Z[158:254]`), we extract:


$$\text{Crop Window: } \text{IL}[0:200] \times \text{XL}[350:650] \times \text{Z}[0:255] \quad\Longrightarrow\quad (200, 300, 255) = 15,300,000\text{ voxels}$$

### Global Z-Score Normalization (No Per-Slice Scaling)

Seismic amplitudes are normalized using a **single global mean and standard deviation** computed strictly on the training inline block ($\mu = 0.0014$, $\sigma = 0.2252$). Per-slice normalization is intentionally avoided because rescaling individual inlines or time slices destroys the true vertical and lateral acoustic impedance contrast between high-amplitude Rijnland Chalk / Zechstein evaporites and low-amplitude North Sea marine shales.

### Strict Inline-Block Split (Preventing Spatial Leakage)

Adjacent voxels in 3D seismic volumes share wavelet sidelobes and lateral reflector continuity over tens of meters. A random voxel `train_test_split` places neighboring voxels of the same seismic trace into both train and test sets, producing artificially inflated accuracy ($>97\%$) via spatial memorization rather than geological generalization.

We enforce a strict **Inline-Block Split** along Axis 0:

* **Training Block (`IL[0:150]`):** `8,907,989` labeled voxels (`y > 0`)
* **Unseen Test Block (`IL[150:200]`):** `2,776,931` labeled voxels (`y > 0`)

---

## 3. Physics-Driven 3D Seismic Attribute Suite

All attributes are computed on the **full contiguous 3D crop with reflective boundary padding (`mode='reflect'`)** prior to spatial splitting or masking (`src/attributes.py`). Applying 1D Hilbert transforms or 3D spatial filters to sparse/masked arrays would introduce severe phase discontinuities and edge artifacts.

| Channel | Attribute | Formula / Window | Geological Rationale |
| --- | --- | --- | --- |
| `0` | **Seismic Amplitude** | $s(x, y, t)$ (global z-score) | Raw acoustic impedance contrast across bedding interfaces |
| `1` | **Envelope** | $E(t) = \Vert{}s(t) + j\,\mathcal{H}\{s(t)\}\Vert{}$ | Reflection strength; separates high-energy chalk/salt packages from quiet shales |
| `2` | **Instantaneous Phase** | $\phi(t) = \arg(s(t) + j\,\mathcal{H}\{s(t)\})$ | Phase-tracks lateral horizon continuity independent of reflection amplitude |
| `3` | **Instantaneous Frequency** | $f(t) = \frac{1}{2\pi}\frac{\partial}{\partial t}\text{unwrap}(\phi(t))$ | Vertical derivative of phase; sensitive to bed thickness and thin-bed tuning |
| `4` | **Sweetness** | $\text{Sw}(t) = \frac{E(t)}{\sqrt{\Vert{}f(t)\Vert{}} + \epsilon}$ | Highlights high-amplitude, low-frequency bodies (sand/carbonate packages) |
| `5` | **Windowed RMS** | $\sqrt{\frac{1}{W_z}\sum_{\tau} s(x,y,t+\tau)^2}$, $W_z = 9$ | Vertical energy integration over a 36 ms window |
| `6` | **3D Local Variance** | $\mathbb{E}_{3\times 3\times 3}[s^2] - (\mathbb{E}_{3\times 3\times 3}[s])^2$ | 3D trace-to-trace discontinuity; highlights fault zones, unconformities, and chaotic salt flanks |

*(An 8th vectorized 3D GLCM Contrast attribute is also provided in `src/attributes.py` via `compute_glcm_contrast_3d`.)*

---

## 4. Model Evaluation & Feature Importance

Trained `RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)` on the `8,907,989` training-block voxels and evaluated on the `2,776,931` unseen test-block voxels (`IL[150:200]`):

* **Strict Spatial Block Accuracy:** `81.24%`
* **Weighted-Average F1:** `0.81`
* **Macro-Average F1 (Classes 1–5):** `0.44` (`Macro Precision = 0.49`, `Macro Recall = 0.46`)

### Per-Class Performance (`IL[150:200]`)

| Class | Stratigraphic Unit | Precision | Recall | F1-Score | Support (Voxels) |
| --- | --- | --- | --- | --- | --- |
| `1` | **Middle North Sea Group** | `0.68` | `0.55` | `0.61` | `374,219` |
| `2` | **Lower North Sea Group** | `0.89` | `0.92` | `0.91` | `2,081,322` |
| `3` | **Rijnland / Chalk Group** | `0.50` | `0.62` | `0.56` | `197,220` |
| `4` | **Scruff Group** | `0.34` | `0.04` | `0.07` | `113,041` |
| `5` | **Zechstein Group** | `0.03` | `0.19` | `0.05` | `11,129` |
| **All** | **Macro Average** | **`0.49`** | **`0.46`** | **`0.44`** | **`2,776,931`** |
| **All** | **Weighted Average** | **`0.81`** | **`0.81`** | **`0.81`** | **`2,776,931`** |

### Ranked Feature Importances

| Rank | Feature Channel | Gini Importance |
| --- | --- | --- |
| **1** | **3D Local Variance (`window=3×3×3`)** | **22.94%** |
| **2** | **Windowed RMS Amplitude (`window_z=9`)** | **21.72%** |
| **3** | **Sweetness** | **15.20%** |
| **4** | **Instantaneous Phase** | **13.80%** |
| **5** | **Instantaneous Frequency** | **10.90%** |
| **6** | **Envelope** | **9.44%** |
| **7** | **Raw Seismic Amplitude** | **6.00%** |

| Normalized Confusion Matrix | Feature Importance Ranking |
| --- | --- |
|  |  |

---

## 5. Geological Interpretation & Limitations

The Random Forest separates the thick, laterally continuous Cenozoic clastic packages cleanly: **Lower North Sea Group (Class 2, `F1 = 0.91`)**, **Middle North Sea Group (Class 1, `F1 = 0.61`)**, and the high-impedance **Rijnland / Chalk Group (Class 3, `F1 = 0.56`)** are well resolved because their contrasting multi-sample acoustic energy and internal reflector continuity are captured directly by the top two features, **3D Local Variance (`22.94%`)** and **Windowed RMS (`21.72%`)**.

Conversely, misclassification is concentrated in the deepest Mesozoic/Paleozoic units—**Scruff Group (Class 4, `F1 = 0.07`)** and **Zechstein Group (Class 5, `F1 = 0.05`)**—which are predominantly confused with the overlying **Lower North Sea (Class 2)** and **Rijnland / Chalk (Class 3)** units. Three physical and structural factors explain this behavior:

1. **Thin-Bed Tuning Below Seismic Resolution:** Along the structural highs and diapir flanks in `IL[150:200]`, the Scruff and Rijnland intervals thin toward the quarter-wavelength tuning thickness ($\lambda/4$), causing wavelet interference that blends their instantaneous frequency and amplitude signatures with adjacent formations.
2. **Spatial Pinch-Out Across the Inline Split:** The Zechstein salt dome crests in `IL[0:150]` (`726k` voxels) and plunges/pinches out rapidly across the unseen test block `IL[150:200]` (only `11,129` voxels, `<0.4%` of test voxels), creating severe local class shift across the spatial boundary.
3. **Point-wise Receptive Field Limitation:** Even with `3×3×3` variance and 9-sample vertical RMS windows, a voxel-wise tree classifier lacks the long-range structural context (e.g., stratigraphic ordering from seabed to salt base) that a human interpreter or multi-scale 3D encoder uses to track unconformities.

---

## 6. Quickstart & Interactive Streamlit Web App

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate facies-ml

```

### 2. Data Setup

Download the F3 benchmark arrays from [Zenodo Record 3755060](https://zenodo.org/record/3755060) and place `train_seismic.npy` and `train_labels.npy` inside `data/raw/train/`. Then run Notebooks `01` through `05` in order inside `notebooks/`.

### 3. Launch the Interactive Web Application

```bash
streamlit run app.py

```

Upload `data/test_slice_175.npy` (or any `(300, 255, 7)` inline feature slice `.npy`) to inspect the side-by-side **Seismic Amplitude**, **Ground Truth Facies**, and **Random Forest Prediction**, view live per-slice Accuracy and Macro-F1, and download the predicted facies map (`predicted_facies.npy`).

```

```
