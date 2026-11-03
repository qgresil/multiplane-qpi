"""Staircase α calibration (SI §4) on simulated data: a thin etched-glass staircase in air imaged with the SI forward model
(qpi.transfer_function + weak object Γ = i φ̂ H, I = 1 + 2·scale·Re Γ, camera offset added) must give back the α that undoes
`scale`; the jump definition, the height mapping and the acquisition checks are tested on their own."""
import numpy as np
import pytest

from mpqpi import qpi, io, staircase as st
from mpqpi.params import REG

PX, BAND, N_ABOVE, N_GLASS, LAMBDA0 = 0.111, (0.5445, 0.6195), 1.0, 1.51, 0.58
DZ, NZ, OFFSET, COUNTS = 0.2, 50, 100.0, 1000.0
K0_DN = qpi.k0_dn_h(1.0, N_GLASS, N_ABOVE, LAMBDA0)
STEPS_NM = {"outer L": 37.5, "long line": 51.5, "inner L": 68.5}       # the user's AFM heights
# geometry, in pixels: a 2.7 µm wide stripe (two long horizontal edges), an outer square and an inner square on top of it
EXPECTED = {"H1": ("long line", 10), "H2": ("long line", 34), "H3": ("outer L", 50), "H4": ("inner L", 85),
            "H5": ("inner L", 135), "H6": ("outer L", 170), "V1": ("outer L", 50), "V2": ("inner L", 85),
            "V3": ("inner L", 135), "V4": ("outer L", 170)}
EDGE_KW = dict(smooth_sigma_px=1.0, canny_sigma_px=1.5, canny_quantiles=(0.85, 0.93), min_length_um=4.0, merge_angle_deg=4.0,
               merge_dist_um=0.7, merge_gap_um=1.5, hough_threshold=10, hough_gap_um=1.0, border_px=5)
PROF_KW = dict(half_um=1.5, end_margin_um=1.0)
WINDOW_UM, SEGMENT_UM = 0.8, 3.0


@pytest.fixture(autouse=True)
def clean_registry():
    REG.clear()
    yield


def _height_map(n=192):
    h = np.zeros((n, n))
    h[50:170, 50:170] += 1e-3 * STEPS_NM["outer L"]
    h[85:135, 85:135] += 1e-3 * STEPS_NM["inner L"]
    h[10:34, 10:182] += 1e-3 * STEPS_NM["long line"]
    return h


@pytest.fixture(scope="module")
def model():
    tf = qpi.transfer_function(0.95, 0.26, N_ABOVE, BAND, n_samples=1_000_000)
    gzc = qpi.gz_cutoff(0.26, BAND, N_ABOVE, "si_text")
    phi = K0_DN * _height_map()
    return tf, gzc, phi


def _counts(model, scale, z_shift=0.0):
    """Simulated camera frames of a stage z-stack (offset included), as io.load would return them."""
    tf, _, phi = model
    z = (np.arange(NZ) - NZ // 2) * DZ + z_shift
    I, _ = st.simulate_thin_object(phi, tf, z, PX, scale)
    return COUNTS * I + OFFSET


def _pipeline(model, frames):
    """Notebook 03 method: offset, flat field, retrieval, edges, mapping, α per edge, combination."""
    tf, gzc, _ = model
    sig = io.to_signal(frames, OFFSET)
    G = st.retrieve_channel(sig, st.flat_field(sig, 90.0), DZ, PX, tf, +1, gzc)
    kf = st.focus_plane(G, 8.0)
    edges = st.detect_edges(G[kf].imag, PX, **EDGE_KW)
    edges = [st.recentre(G[kf], e, PX, block_um=1.0, **PROF_KW) for e in edges]
    edges = st.merge_edges(edges, PX, 4.0, 0.7, 1.5)
    edges = st.split_at_crossings(edges, PX, gap_um=1.5, reach_um=1.5, min_length_um=4.0)
    edges = st.label_edges(edges, 20.0)
    used, _ = st.assign_heights(edges, {k: v[0] for k, v in EXPECTED.items()}, STEPS_NM)
    res = [st.measure_edge(G, e, h, K0_DN, PX, window_um=WINDOW_UM, segment_um=SEGMENT_UM, z_upsample=8, **PROF_KW)
           for e, _, h in used]
    return G, edges, used, res


def _oracle_alpha(model, used, scale):
    """α from the ideal Γ₊ (infinite z range, no mirroring) of the same model, same jump definition."""
    tf, gzc, phi = model
    Gi = scale * st.ideal_gamma_plus(phi, tf, PX, gzc)
    return np.array([st.solve_alpha(*st.edge_profile(Gi, e, PX, **PROF_KW), K0_DN * h, WINDOW_UM) for e, _, h in used])


def test_edges_found_and_labelled(model):
    _, edges, _, _ = _pipeline(model, _counts(model, 1 / 3.15))
    by = {e["label"]: e for e in edges}
    assert set(EXPECTED) <= set(by)
    for lab, (_, pos) in EXPECTED.items():
        c = (by[lab]["p0"] + by[lab]["p1"]) / 2
        assert abs(c[1 if lab[0] == "H" else 0] - (pos - 0.5)) < 1.0, lab        # step between pixels pos − 1 and pos


def test_alpha_recovers_known_alpha(model):
    """Calibrate the model so that the ideal retrieval needs α = 1 at scale 1; simulate the scale 1/3.15 -> α must be 3.15."""
    _, _, used, _ = _pipeline(model, _counts(model, 1.0))
    kappa = np.median(_oracle_alpha(model, used, 1.0))                     # ideal-system α at scale 1
    alpha_true = 3.15
    _, _, used, res = _pipeline(model, _counts(model, kappa / alpha_true))
    a = np.array([r["alpha"] for r in res])
    m, u = st.combine(a, [r["alpha_se"] for r in res], "mean")
    assert np.all(np.isfinite(a))
    print(f"alpha_true {alpha_true}, recovered {m:.3f} ± {u:.3f}, per edge {np.round(a, 3)}")
    assert abs(m / alpha_true - 1) < 0.06                                  # finite 10 µm stack + mirroring: ≈ +3 %
    assert a.std() / a.mean() < 0.04                                       # same α for 37.5, 51.5 and 68.5 nm steps


def test_alpha_undoes_the_scale_factor(model):
    out = {}
    for scale in (1 / 3.15, 1 / 6.3):
        _, _, used, res = _pipeline(model, _counts(model, scale))
        a = np.array([r["alpha"] for r in res])
        assert np.allclose(a / _oracle_alpha(model, used, scale), 1, atol=0.07)
        out[scale] = st.combine(a, [r["alpha_se"] for r in res], "inverse_variance")[0]
    assert abs(out[1 / 6.3] / out[1 / 3.15] - 2) < 0.06                    # α ∝ 1/scale (weak object, eq. S28 inverted exactly)


def test_alpha_independent_of_surface_position_between_planes(model):
    """With z interpolation of the edge profiles, α does not depend on where the surface falls between two 200 nm planes;
    without it (z_upsample = 1) it does, by several %."""
    tf, gzc, phi = model
    e = st.label_edges([st._seg((30, 33.5), (160, 33.5))], 20.0)[0]
    out = {1: [], 8: []}
    for shift in (0.0, 0.05, 0.1):
        sig = io.to_signal(_counts(model, 1 / 3.15, shift), OFFSET)
        G = st.retrieve_channel(sig, st.flat_field(sig, 90.0), DZ, PX, tf, +1, gzc)
        for up in out:
            out[up].append(st.measure_edge(G, e, 0.0515, K0_DN, PX, window_um=WINDOW_UM, segment_um=SEGMENT_UM, z_upsample=up,
                                           **PROF_KW)["alpha"])
    assert np.ptp(out[8]) / np.mean(out[8]) < 0.01
    assert np.ptp(out[1]) / np.mean(out[1]) > 0.03


def test_offset_must_be_subtracted(model):
    """Forgetting the camera offset biases Γ (and α) by (I − offset)/I (DESIGN rule 5)."""
    frames = _counts(model, 1 / 3.15)
    tf, gzc, _ = model
    G_ok = st.retrieve_channel(io.to_signal(frames, OFFSET), st.flat_field(io.to_signal(frames, OFFSET), 90.0), DZ, PX, tf, +1, gzc)
    G_bad = st.retrieve_channel(io.to_signal(frames, 0), st.flat_field(io.to_signal(frames, 0), 90.0), DZ, PX, tf, +1, gzc)
    r = np.abs(G_bad).max() / np.abs(G_ok).max()
    assert abs(r - COUNTS / (COUNTS + OFFSET)) < 0.01


def test_jump_and_solve_alpha_analytic():
    s = np.linspace(-1.5, 1.5, 241); w = 0.3; a_true = 2.0
    phi = 0.2 * (s / w) * np.exp(-(s / w) ** 2)                            # bipolar edge, maximum on the + side
    prof = (np.exp(1j * phi) - 1) / a_true                                 # eq. S28 with α = 2, I₀ = 1, gives exactly φ
    expected = 2 * 0.2 * np.exp(-0.5) / np.sqrt(2)
    assert abs(st.jump(s, prof, a_true, 0.8) - expected) < 1e-3
    assert st.jump(s[::-1], prof[::-1], a_true, 0.8) == pytest.approx(st.jump(s, prof, a_true, 0.8))   # sign follows s
    assert st.jump(-s, prof, a_true, 0.8) == pytest.approx(-st.jump(s, prof, a_true, 0.8))
    assert st.solve_alpha(s, prof, expected, 0.8) == pytest.approx(a_true, rel=1e-3)
    assert np.isnan(st.solve_alpha(s, prof, 10.0, 0.8))                    # unreachable target
    G = 0.05 + 0.1j
    a = st.alpha_single_point(G, 0.4)
    assert qpi.phase(np.array([G]), a)[0] == pytest.approx(0.4, abs=1e-6)


def test_assign_heights_mapping():
    edges = st.label_edges([st._seg((0, 10), (100, 10)), st._seg((50, 0), (50, 100)), st._seg((0, 60), (100, 61))], 20.0)
    assert [e["label"] for e in edges] == ["H1", "H2", "V1"] and edges[2]["orientation"] == "vertical"
    used, unassigned = st.assign_heights(edges, {"V1": "inner L", "H2": "outer L"}, STEPS_NM)
    assert [(e["label"], n, round(h, 4)) for e, n, h in used] == [("V1", "inner L", 0.0685), ("H2", "outer L", 0.0375)]
    assert unassigned == ["H1"]
    with pytest.raises(KeyError):
        st.assign_heights(edges, {"V7": "inner L"}, STEPS_NM)             # label not detected
    with pytest.raises(KeyError):
        st.assign_heights(edges, {"V1": "middle"}, STEPS_NM)              # step not in the AFM list
    with pytest.raises(ValueError):
        st.assign_heights(edges, {"V1": "x"}, {"x": -3})


def test_split_at_t_junction():
    h = st._seg((0, 50), (200, 50)); v = st._seg((100, 52), (100, 150))  # vertical edge ending on the horizontal one
    pieces = st.split_at_crossings([h, v], 0.1, gap_um=1.0, reach_um=1.0, min_length_um=2.0)
    hs = sorted([p for p in pieces if abs(p["u"][0]) > 0.9], key=lambda p: p["p0"][0] + p["p1"][0])
    assert len(hs) == 2 and max(hs[0]["p0"][0], hs[0]["p1"][0]) == pytest.approx(90)
    assert len([p for p in pieces if abs(p["u"][1]) > 0.9]) == 1          # the vertical one is not cut (junction at its end)


def test_check_zstack():
    gzc = qpi.gz_cutoff(0.26, BAND, 1.0, "si_text")
    w, _ = st.check_zstack(None, 50, 0.2, gzc)
    assert "not a stage z-stack" in w[0]
    w, dgz = st.check_zstack(np.arange(50) * 0.2, 50, 0.2, gzc)
    assert w == [] and dgz < gzc
    w, dgz = st.check_zstack(np.arange(21) * 0.05, 50, 0.2, gzc)        # the user's 21 × 50 nm stack
    assert len(w) == 2 and dgz > gzc
