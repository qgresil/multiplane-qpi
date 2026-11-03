# multiplane-qpi

Calibration and 3D quantitative phase imaging for an 8-plane multiplane microscope, following the methods of
Descloux, Grußmayer et al., *Combined multi-plane phase retrieval and super-resolution optical fluctuation imaging for 4D cell
microscopy*, Nat. Photonics 12, 165 (2018) and its Supplementary Information (SI).

Every number used by the pipeline is declared with its origin — `paper` (SI section/equation), `measured`, `manufacturer`,
`derived`, `user` or **`heuristic`** (an implementation choice) — and every notebook prints the list, so that assumptions are
visible and reviewable (see `docs/DESIGN.md`).

## Pipeline (one session = run in this order)

| Notebook | Sample / acquisition | Produces | SI |
|---|---|---|---|
| `01_registration_plane_spacing` | fluorescent beads, stage z-stack (200 nm steps) | `calibration.json`: panels, plane offsets Δ and spacing, affines, channel transmission | §6.1–6.4 |
| `02_psf_uipsf` *(extension)* | same bead stack | PSF / pupil characterisation with uiPSF | — |
| `03_staircase_alpha` | etched staircase, brightfield z-stack 50 × 200 nm + AFM heights | `alpha.json`: phase calibration factor α | §4 |
| `04_ps_beads_phase` | 200 nm polystyrene beads, brightfield z-stack 50 × 200 nm (+ single-shot movie) | `beads.json`: bead phase vs 0.34 rad, widths vs Fig. S3 | §3–5 |

## Install (macOS or Windows)

```
git clone <this repo> && cd multiplane-qpi
conda env create -f environment.yml && conda activate mpqpi
cp config/local.example.yaml ~/.mpqpi.yaml      # then edit data_root (and uipsf_python for notebook 02)
python tools/py2nb.py                            # generates notebooks/*.ipynb from the .py sources
pytest -q                                        # tests on simulated data
jupyter lab notebooks/
```
uiPSF (notebook 02 only) needs its own environment, see `environment-uipsf.yml`.

## Data
Raw data stay outside the repository. `.dat` + `.info` files (LabVIEW acquisition) and TIFF folders with `acquisition.json`
(Python controller) are both read by `mpqpi.io.load`. Paths in notebooks are relative to `data_root`. Each notebook writes its
results next to the input data in `<stem>_<NN>_<step>/`, with a provenance block (code version, all parameters with their tags,
inputs, computer, date).

## Instrument
`instruments/lob_multiplane.yaml` holds the instrument description. When you measure something (pixel size, NA_ill, camera
offset ...), change its tag to `measured` and say how in `source`.
