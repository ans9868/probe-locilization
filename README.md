# Structured Probe Localization: Statistical Priors for Invasive BCI Electrode Placement Verification

## Motivation

Accurate electrode placement is critical for invasive BCI performance. Current approaches use deep learning to classify probe location from neural recordings against a whole-brain atlas — treating it as a 100+ class problem over all possible brain regions. This is computationally expensive and ignores the structure of the actual clinical problem:

- Surgeons know the **target region** and its **neighbors** — it's a 4-5 class problem, not 100
- Probe misplacement errors follow **predictable spatial patterns** (drift along insertion axis, lateral deviation)
- **Depth estimation** along the probe shank is a 1D regression problem, not a classification problem

We argue that incorporating spatial priors and reducing the problem to its clinically relevant structure yields faster, more reliable localization than brute-force deep learning over the whole brain.

## Research Questions

1. **Depth estimation**: Given a known target region, can we estimate probe depth from streaming neural features (firing rates, LFP power spectra, waveform shapes) using statistical models?

2. **Local region disambiguation**: Given 4 adjacent candidate regions, can a lightweight classifier with spatial priors outperform a whole-brain deep learning classifier?

3. **Failure mode detection**: What are the most common electrode misplacement patterns in invasive BCI, and can we build targeted detectors for each specific failure mode?

4. **Dimensionality reduction for real-time verification**: Can sparse random projections (JL) preserve the discriminative structure needed for probe localization, enabling real-time verification on edge hardware during surgery?

## Methods

### Approach 1: Statistical Depth Estimation
- Model neural feature gradients along the insertion axis (layer-specific firing rates, oscillation frequencies)
- Use Bayesian inference with anatomical priors (cortical layer thickness distributions) to estimate depth
- Compare: linear regression, Gaussian process regression, vs. MLP baseline

### Approach 2: Structured Local Classification
- Reduce the problem from whole-brain (100+ classes) to local neighborhood (4-5 classes)
- Incorporate spatial adjacency priors as a structured prior in the classifier
- Compare: logistic regression with spatial features, random forest with anatomical priors, vs. ResNet on raw waveforms (baseline)

### Approach 3: Sparse Projections for Real-Time Verification
- Apply Sparse JL projections to high-dimensional neural snapshots (reframing the financial market snapshot framework)
- Neural "market snapshot" = concatenated multi-channel features over a time window
- Test whether JL-compressed features preserve local discriminability for the 4-class problem
- Compare: dense Gaussian JL, Sparse JL (SJLT), PCA, and raw features

### Connection to Financial Market Snapshot Framework
The financial snapshot approach (Cohen-Jayram-Nelson SJLT analysis) maps directly:

| Financial Domain | BCI Domain |
|---|---|
| N = 500 assets | N = 384 Neuropixels channels |
| Tₘ = 252 trading days | Tₘ = 1000 time bins (1 second at 1kHz) |
| D = N × Tₘ = 126,000 | D = N × Tₘ = 384,000 |
| Latent factors (market, sector) | Latent factors (brain state, movement intention) |
| Regime detection (bull/bear/crisis) | Region detection (M1/S1/premotor/SMA) |
| Low effective rank (r ≈ 10-50) | Low effective rank (neural manifold dim ≈ 10-30) |

The same theoretical question applies: does SJLT distortion improve when the data lives near a low-rank manifold, and does this enable real-time probe verification on edge hardware?

## Datasets

- **IBL Brain Wide Map** — Neuropixels recordings with known anatomical locations (ground truth)
- **Allen Brain Atlas Neuropixels Visual Coding** — well-characterized probe trajectories through visual cortex layers
- **Steinmetz et al. 2019** — multi-region Neuropixels recordings with histological verification
- **Synthetic** — spiked covariance model mimicking neural population structure with known region labels

## Evaluation Metrics

- **Depth estimation error** (mm) — MAE and std across probes
- **Local classification accuracy** — 4-class accuracy vs. whole-brain accuracy
- **Failure mode detection** — precision/recall for specific misplacement types
- **Compression ratio vs. accuracy** — quality under JL projection at various target dimensions
- **Inference latency** — time per classification on edge hardware

## Project Structure

```
├── data/
│   ├── ibl_loader.py           # IBL Brain Wide Map
│   ├── allen_loader.py         # Allen Neuropixels Visual Coding
│   └── steinmetz_loader.py     # Steinmetz 2019
├── features/
│   ├── neural_snapshots.py     # Multi-channel feature extraction
│   ├── depth_features.py       # Layer-specific feature gradients
│   └── spatial_priors.py       # Anatomical adjacency graphs
├── models/
│   ├── depth_estimator.py      # Bayesian depth regression
│   ├── local_classifier.py     # 4-class structured classifier
│   ├── global_baseline.py      # Whole-brain DL baseline
│   └── failure_detector.py     # Targeted misplacement detector
├── projections/
│   ├── sparse_jl.py            # SJLT for neural snapshots
│   ├── pca.py                  # PCA baseline
│   └── distortion_analysis.py  # Low-rank distortion bounds
├── experiments/
│   ├── run_depth.py
│   ├── run_local_vs_global.py
│   ├── run_compression.py
│   └── configs/
├── notebooks/
├── results/
└── README.md
```

## Timeline

- **Day 1-2**: Data pipeline (IBL/Allen downloads, feature extraction, ground truth alignment)
- **Day 3**: Implement depth estimator + local classifier + global baseline
- **Day 4**: Implement JL projection framework (port from financial snapshot code)
- **Day 5**: Run experiments — depth estimation, local vs. global, compression sweeps
- **Day 6**: Analysis, figures, failure mode characterization
- **Day 7**: Write-up and polish

## Expected Contributions

1. **Methodological**: Show that structured statistical approaches outperform brute-force DL for probe localization when clinical priors are available
2. **Practical**: A lightweight, real-time probe verification tool that works on edge hardware during surgery
3. **Theoretical**: Extend the SJLT low-rank analysis to neural population data, characterizing when random projections preserve local discriminability

## References

- Cohen, Jayram, Nelson (2018). Simple Analyses of the Sparse Johnson-Lindenstrauss Transform. SOSA.
- Steinmetz et al. (2019). Distributed coding of choice, action and engagement across the mouse brain. Nature.
- IBL et al. (2023). Reproducible Brain-Wide Association Study. eLife.
- Jun et al. (2017). Fully integrated silicon probes for high-density recording of neural activity. Nature.
- Liberty (2013). Simple and Deterministic Matrix Sketching. KDD.

## Author

Adel Sahuc — NYU Tandon School of Engineering  
ans9868@nyu.edu
