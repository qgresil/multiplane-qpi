"""Bead validation on simulated data (SI §3.2 forward model): detection, per-bead peak phase vs α, Fourier re-centring of the
bead average, single-shot assembly (plane order and spacing) from a fake calibration.json."""
import json
import numpy as np
import pytest
from scipy.spatial import cKDTree

from mpqpi import qpi, beads
from mpqpi.params import REG

BAND, NA_DET, NA_ILL, N_MED = (0.5445, 0.6195), 1.2, 0.26, 1.33
DZ, PX = 0.2, 0.111


@pytest.fixture(autouse=True)
def clean_registry():
    REG.clear()
    yield


@pytest.fixture(scope="module")
def tf():
    return qpi.transfer_function(NA_DET, NA_ILL, N_MED, BAND, n_samples=1_000_000)


@pytest.fixture(scope="module")
def gzc():
    return qpi.gz_cutoff(NA_ILL, BAND, N_MED, "si_text")


def _bead_field(n, ny, nx, nz, rng, edge_px=12, min_sep_px=14, z_margin=8):
    pos = []
    while len(pos) < n:
        p = np.array([rng.uniform(z_margin, nz - 1 - z_margin) * DZ, rng.uniform(edge_px, ny - edge_px) * PX,
                      rng.uniform(edge_px, nx - edge_px) * PX])
        if all(np.hypot(*(p[1:] - q[1:])) > min_sep_px * PX for q in pos):
            pos.append(p)
    return np.array(pos)


def _reference_peak(tf, gzc, alpha, amp):
    """Continuous peak phase of one isolated, voxel-centred, noise-free bead: the value the pipeline must recover."""
    I = beads.simulate_stack(tf, (50, 64, 64), DZ, PX, [[25 * DZ, 32 * PX, 32 * PX]], +1, amp)
    G = qpi.cross_spectral_density(I, DZ, PX, tf, +1, gzc); phi = qpi.phase(G, alpha)
    return beads.peak_fourier(phi, *np.unravel_index(np.argmax(phi), phi.shape), (8, 8, 8), field=G, alpha=alpha)[0]


def test_detection_and_peak_phase_scale_with_alpha(tf, gzc):
    rng = np.random.default_rng(1)
    nz, ny, nx, amp = 50, 128, 128, 0.07                                  # α = 3.15: peak ≈ 0.3 rad, as for the SI beads
    pos = _bead_field(7, ny, nx, nz, rng)
    I = beads.simulate_stack(tf, (nz, ny, nx), DZ, PX, pos, +1, amp)
    I = I + rng.normal(0, 5e-3, I.shape).astype(np.float32)
    assert qpi.choose_sign(I, DZ, PX, tf, gzc, 3.15) == +1
    G = qpi.cross_spectral_density(I, DZ, PX, tf, +1, gzc)
    for alpha in (1.0, 3.15):
        phi = qpi.phase(G, alpha)
        xy = beads.detect(phi.max(0), k_sigma=8, footprint_px=9, edge_px=4)
        assert len(xy) == len(pos)                                         # every bead, no false detection
        t = beads.measure_beads(phi, xy, 2, "fourier", 10, 3, (8, 8, 8), 0.0, G, alpha)
        d, j = cKDTree(pos[:, [2, 1]] / PX).query(np.c_[t["x"], t["y"]])
        assert d.max() < 0.5                                               # sub-pixel lateral position
        assert np.abs(t["k"] - pos[j, 0] / DZ).max() < 0.5                 # sub-frame focus
        ref = _reference_peak(tf, gzc, alpha, amp)
        assert abs(np.mean(t["peak"]) / ref - 1) < 0.02                    # recovers the simulated value
        assert np.mean(t["peak_max_voxel"]) < np.mean(t["peak"])            # (the max voxel misses the off-grid peak)
        if alpha == 1.0:
            p1 = np.mean(t["peak"])
        else:                                                              # S28: tan φ ≈ α Im Γ₊ (Re Γ₊ small at focus)
            assert abs(np.tan(np.mean(t["peak"])) / np.tan(p1) / alpha - 1) < 0.1
            a = beads.alpha_from_beads(ref, t["re_g"], t["im_g"])          # target = noise-free model peak at this α
            assert abs(np.median(a) / alpha - 1) < 0.05                    # S28 inversion returns the α used


def test_fourier_recentring_does_not_broaden(tf, gzc):
    rng = np.random.default_rng(2)
    nz, ny, nx = 50, 160, 160
    pos = _bead_field(10, ny, nx, nz, rng, edge_px=20, min_sep_px=24, z_margin=12)
    phi = qpi.phase(qpi.cross_spectral_density(beads.simulate_stack(tf, (nz, ny, nx), DZ, PX, pos, +1, 0.02),
                                               DZ, PX, tf, +1, gzc), 3.15)
    t = beads.measure_beads(phi, beads.detect(phi.max(0), 8, 9, 4), 2, "fourier", 10, 3, (8, 8, 8))
    half = (8, 10, 10)
    avg, n, _ = beads.average_beads(phi, np.c_[t["k"], t["y"], t["x"]], half, margin=3)
    assert n == len(pos)
    single = beads.simulate_stack(tf, (nz, 64, 64), DZ, PX, [[25 * DZ, 32 * PX, 32 * PX]], +1, 0.02)
    ref = beads.centred_crop(qpi.phase(qpi.cross_spectral_density(single, DZ, PX, tf, +1, gzc), 3.15), half)
    f_avg, f_ref = beads.fwhm_xyz(avg, DZ, PX), beads.fwhm_xyz(ref, DZ, PX)
    for a in "xyz":
        assert abs(f_avg[a] / f_ref[a] - 1) < 0.05, (a, f_avg, f_ref)
    assert abs(avg.max() / ref.max() - 1) < 0.05
    # integer-voxel crops (no re-centring) do broaden: the test is sensitive
    naive, _, _ = beads.average_beads(phi, np.round(np.c_[t["k"], t["y"], t["x"]]), half, margin=3)
    assert beads.fwhm_xyz(naive, DZ, PX)["x"] > f_avg["x"]


def test_peak_fourier_recovers_off_grid_peak(tf, gzc):
    alpha, amp = 3.6, 0.08                                                 # non-linear regime of S28 (peak ≈ 0.3 rad)
    ref = _reference_peak(tf, gzc, alpha, amp)
    for off in ((0.5, 0.0, 0.0), (0.3, -0.4, 0.5), (-0.5, 0.5, 0.5)):
        I = beads.simulate_stack(tf, (50, 64, 64), DZ, PX, [[(25 + off[0]) * DZ, (32 + off[1]) * PX, (32 + off[2]) * PX]], +1, amp)
        G = qpi.cross_spectral_density(I, DZ, PX, tf, +1, gzc); phi = qpi.phase(G, alpha)
        k, y, x = np.unravel_index(np.argmax(phi), phi.shape)
        v, kf, yf, xf, ok, _ = beads.peak_fourier(phi, k, y, x, (8, 8, 8), field=G, alpha=alpha)
        assert ok and abs(v / ref - 1) < 0.01 and phi.max() / ref < 0.97     # max voxel is biased, interpolation is not
        assert np.allclose([yf - 32, xf - 32], off[1:], atol=0.05)          # (axially the S28 peak sits ≈ 0.1 frame off focus)
        assert abs(kf - 25 - off[0]) < 0.3


def _fake_calibration(tmp_path, delta, shifts):
    W = 32
    planes = []
    for i, (d, s) in enumerate(zip(delta, shifts)):
        planes.append({"plane": f"P{i + 1}", "stream": "AB"[i % 2], "panel_id": i, "orientation": "none",
                       "window_raw": {"top": 0, "left": (i // 2) * W, "height": W, "width": W}, "delta_z_um": d,
                       "affine_from_reference": [[1, 0, s[0]], [0, 1, s[1]]], "transmission": 1.0})
    cal = {"planes": planes, "reference_plane": "P1", "plane_order": [p["plane"] for p in planes],
           "plane_spacing_um": {"mean": 0.35, "sd": 0.0, "per_gap": []}, "pixel_um": PX}
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    return json.loads((tmp_path / "calibration.json").read_text()), W


def test_single_shot_order_and_spacing(tmp_path):
    true_sp = 0.35
    # label order is not Δ order (P4 and P5 swapped) and P8's Δ is off by 0.25 µm
    rank = np.array([0, 1, 2, 4, 3, 5, 6, 7])
    delta = true_sp * rank + 0.1; delta[7] += 0.25
    shifts = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (-1.0, 1.0), (2.0, -1.0), (0.0, 0.0), (1.0, 1.0), (-2.0, 0.0)]
    cal, W = _fake_calibration(tmp_path, delta.tolist(), shifts)
    fit = beads.plane_spacing_fit([p["delta_z_um"] for p in cal["planes"]])
    assert abs(fit["spacing"] - true_sp) < 0.02                            # robust fit of sorted Δ
    assert abs(np.mean(np.diff(delta)) - true_sp) > 0.03                   # the old mean(diff) in label order is biased
    assert list(fit["order"]) == [0, 1, 2, 4, 3, 5, 6, 7]
    # each panel: background 1 + a spot at reference (16, 12) moved by its affine, value tagging the plane
    frames = {"A": np.ones((W, 4 * W), np.float32), "B": np.ones((W, 4 * W), np.float32)}
    for i, (p, s) in enumerate(zip(cal["planes"], shifts)):
        x, y = 16 + s[0], 12 + s[1]
        frames[p["stream"]][int(y), (i // 2) * W + int(x)] = 2 + i
    flats = [np.ones((W, W), np.float32)] * 8
    vol, labels, valid = beads.assemble_single_shot(frames, cal["planes"], flats, "stage_focus")
    assert labels == ["P8", "P7", "P6", "P4", "P5", "P3", "P2", "P1"]     # descending Δ = increasing z-scan stage order
    for j, lab in enumerate(labels):
        i = int(lab[1:]) - 1
        assert abs(vol[j, 12, 16] - (2 + i)) < 1e-5                       # registered onto the reference pixel
    assert not valid.all() and valid[12, 16]
    vol2, labels2, _ = beads.assemble_single_shot(frames, cal["planes"], flats, "sample_depth")
    assert labels2 == labels[::-1]


def test_single_shot_frame_puts_bead_mid_volume():
    delta = np.array([0.0, 0.35, 0.7, 1.05, 1.4, 1.75, 2.1, 2.45])
    k = beads.single_shot_frame(10.0, 0.0, delta, 0.2, "stage_focus")
    assert k == 10 + round(1.225 / 0.2)


def test_read_alpha(tmp_path):
    assert beads.read_alpha(tmp_path / "missing.json") is None
    (tmp_path / "a.json").write_text(json.dumps({"alpha": {"value": 4.2, "sd": 0.3}}))
    assert beads.read_alpha(tmp_path / "a.json") == (4.2, 0.3)
    (tmp_path / "b.json").write_text(json.dumps({"alpha": 3.9}))
    assert beads.read_alpha(tmp_path / "b.json") == (3.9, None)


def test_stats_and_trend():
    s = beads.stats([0.3, 0.32, 0.34, 0.36])
    assert s["n"] == 4 and abs(s["mean"] - 0.33) < 1e-12 and abs(s["median"] - 0.33) < 1e-12
    tr = beads.trend(np.arange(8), 0.3 - 0.01 * np.arange(8), np.full(8, 0.002))
    assert abs(tr["slope"] + 0.01) < 1e-9 and tr["n_decreasing_steps"] == 7


def test_frame_shifts_recover_known_jitter(tf):
    """Extension E3: a simulated bead field moved by known sub-pixel steps between frames."""
    rng = np.random.default_rng(3)
    pos = _bead_field(12, 128, 128, 30, rng, z_margin=3)
    I = beads.simulate_stack(tf, (30, 128, 128), DZ, PX, pos, +1, 0.07)
    jit = np.cumsum(rng.normal(0, 0.3, (30, 2)), 0); jit -= jit[0]
    moved = np.stack([beads.apply_shifts(I[k:k + 1], -jit[k:k + 1])[0] for k in range(30)])     # content moved by +jit
    chans = [moved + rng.normal(0, 3e-3, moved.shape).astype(np.float32) for _ in range(4)]     # 4 channels of one camera
    est = beads.frame_shifts(chans, 5, 2, 3)
    assert np.sqrt(np.mean((np.diff(est, axis=0) - np.diff(jit, axis=0)) ** 2)) < 0.05   # px per step
