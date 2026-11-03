# Design and conventions

## Purpose
A calibration → measurement pipeline for the 8-plane multiplane microscope that follows the methods of Descloux et al.,
Nat. Photonics 12, 165 (2018) and its Supplementary Information (SI). Reference PDF (not in the repo):
`41566_2018_109_MOESM1_ESM.pdf` (SI). **Paper methods first; anything else is an explicitly labelled extension.**

## Pipeline (one notebook per task, run in this order per session)

| # | Notebook | Input | Output (folder next to the data) | SI |
|---|---|---|---|---|
| 01 | `01_registration_plane_spacing` | fluorescent bead z-stack (stage scan, 200 nm steps) | `<stem>_01_registration/calibration.json` | §6.1–6.4, Fig. S16, §8.3 |
| 02 | `02_psf_uipsf` (**extension**) | same bead z-stack + calibration.json | `<stem>_02_psf/` | — (SI §6.3 is a design analysis only) |
| 03 | `03_staircase_alpha` | brightfield staircase z-stack, 50 planes × 200 nm + AFM step heights + calibration.json | `<stem>_03_alpha/alpha.json` | §4, eq. S28, Fig. S4, §8.5 |
| 04 | `04_ps_beads_phase` | brightfield 200 nm PS bead z-stack (50 × 200 nm) and optionally a single-shot movie + calibration.json + alpha.json | `<stem>_04_beads/beads.json` | §3.3, §4, §5, Fig. S3, §8.4 |

α is calibrated on the staircase (03) *before* the bead validation (04), as in SI §4 (the beads validate the calibration:
0.33 ± 0.014 rad measured vs 0.34 rad theory).

## Rules
1. **Every number that can change a result is declared** with `REG.declare(name, value, tag, source, unit, note)`
   (`mpqpi.params`), in the notebook's parameter cell or, for module internals, as an explicit function argument whose value the
   notebook declares. No bare magic numbers inside modules (pure plotting cosmetics excepted). Tags: `paper` (cite SI §/eq./Fig.),
   `measured`, `manufacturer`, `derived` (formula), `user` (sample/acquisition choices), `heuristic` (implementation choice:
   give the rationale). Every notebook calls `REG.banner()` after its parameter cell and stores `REG.to_dict()` in its output.
2. **Instrument facts come from `instruments/lob_multiplane.yaml`** via `Instrument().value(key)` (declares them with their tag).
3. **Sample-agnostic**: the sample index `n_sample` (and immersion/medium indices) are explicit `user` parameters, propagated to
   every computation that needs them. Nothing assumes a particular calibration sample preparation.
4. **Stage z = commanded positions** (`mpqpi.io`), never the stage read-outs.
5. **Camera offset is subtracted** (`io.to_signal`) before any intensity normalisation.
6. **Data paths are relative to `data_root`** (`mpqpi.config.data_path`), set per computer in `~/.mpqpi.yaml`.
7. **Downstream notebooks read upstream outputs** (calibration.json, alpha.json); they never recompute them.
8. **Outputs**: one JSON per step written with `mpqpi.provenance.write_json` (adds code version, parameters with tags, inputs),
   figures as PNG (200 dpi) in the same folder, figure names prefixed by section number.
9. **Extensions** (not in the SI) are allowed when useful, but live in clearly titled sections/notebooks ("Extension — not in the
   paper") and never feed the paper-method results unless the user switches them on explicitly.
10. **Notebook source format**: `notebooks/NN_name.py` in percent format (`# %%` code cells, `# %% [markdown]` text cells);
    `python tools/py2nb.py` generates the `.ipynb`. The first markdown cell states: purpose, SI references, inputs, outputs,
    and the list of extensions used.
11. **Methods code lives in `src/mpqpi/<module>.py`**, notebooks stay thin (parameters, calls, figures, short interpretation).
    Every module function has a docstring citing the SI section it implements, or "extension".
12. **Tests** (`tests/test_<module>.py`) on simulated data: each method must recover known inputs (registration recovers a known
    affine; plane spacing recovers known Δ; α recovers a known α; retrieval returns the SI numbers).

## Core API (`src/mpqpi`)
- `params.REG` — registry (`declare`, `banner`, `to_dict`, `heuristics`).
- `config.Instrument` — `.value(key)`, `.pixel_um(objective)`; `config.data_path(rel)`; `config.output_dir(data_file, step)`;
  `config.local()` (per-computer settings).
- `io.load(path, instrument)` → `Acquisition(streams={"A","B"}: (T,H,W), z_um={"A","B"}: commanded z or None, exposure_s, meta)`;
  handles `.dat`+`.info` and TIFF folders with `acquisition.json`; `io.to_signal(frames, offset)`.
- `geometry.panel(frames_by_stream, plane_entry)`, `geometry.warp(img_or_stack, A_from_ref, order=1)` (bilinear, SI §6.4),
  `geometry.fourier_shift`, `geometry.ORIENT`.
- `qpi` — `transfer_function` (eq. S19, source spectrum), `gz_cutoff` (eq. S24; conventions "si_text" / "s24"), `grids`, `lookup`,
  `cross_spectral_density` (eq. S20–S23 with z mirroring and CTF mask), `phase` (eq. S28), `normalise`, `choose_sign`,
  `k0_dn_h`, `bead_phase_theory`.
- `provenance.write_json(path, payload, step, inputs)`.

## calibration.json (contract between 01 and the other notebooks)
Kept compatible with the previous notebook-02 format:
```
{ "planes": [ {"plane": "P1".."P8", "stream": "A"|"B", "panel_id": ..., "window_raw": {top,left,height,width},
               "orientation": "none"|"lr"|"ud"|"lrud", "delta_z_um": Δ in sample space, "affine_from_reference": 2x3,
               "transmission": relative channel transmission (SI §6.4) }, ... ],
  "reference_plane": "P5", "plane_order": [...], "plane_spacing_um": {"mean":..,"sd":..,"per_gap":[..]},
  "pixel_um": .., "objective": .., "n_sample": .., "n_immersion": .., "config": {..}, "provenance": {..} }
```

Δ convention: `delta_z_um` = stage z at which a bead is in focus in plane p, relative to the reference plane, converted to sample
space (× n_sample / n_immersion, paraxial); `delta_z_stage_um` keeps the stage-space value. Notebook 04 orders the single-shot
planes with this convention ("stage_focus").

## alpha.json (contract between 03 and 04)
```
{ "alpha": {"value": α, "uncertainty": σ_α, "combination": .., "plane": ..}, "edges": [...], "settings": {..}, "provenance": {..} }
```
α is defined as in SI §4: the value for which the phase jump of eq. S28 at each staircase edge equals k₀·Δn·h (AFM height h).
