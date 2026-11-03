"""Registration / plane spacing / transmission on a simulated 8-panel bead z-stack with known windows, orientations, affines,
plane offsets Δ and channel transmissions: the methods of mpqpi.registration must recover them."""
import json
import numpy as np
import pytest

from mpqpi import registration as R, geometry, io
from mpqpi.config import Instrument
from mpqpi.params import REG

PX = 0.111                                         # µm, nominal 60x back-projected pixel (SI §6.1)


@pytest.fixture(autouse=True)
def clean_registry():
    REG.clear()
    yield


def simulate_bead_scan(panel=220, gap=24, top=30, nz=36, step=0.2, n_beads=40, min_sep=36, spacing=0.35, seed=0,
                       b_frame_delay=0, offset=100, bg=30, peak=1500, sigma0=1.4, defocus_px_per_um=3.0, bleach=0.01):
    """Two camera streams (T, H, W) of a bead stage scan through an 8-plane prism. Planes in ascending Δ: P1..P8; camera B holds
    P1, P3, P5, P7 and is mirrored left-right, camera A holds P2..P8. Bead images: normalised Gaussians, σ² = σ0² + (c·defocus)²
    (flux conserved), times the channel transmission and an exponential bleaching per frame; Poisson noise; camera offset.
    b_frame_delay: stream-B frame k recorded at stage z_{k + delay} (frame-pairing error)."""
    rng = np.random.default_rng(seed)
    P = 8
    delta = spacing * np.arange(P) + rng.normal(0, 0.01, P); delta -= delta[0]
    trans = rng.uniform(0.85, 1.15, P); trans /= trans.mean()
    stream = np.array(["B" if r % 2 == 0 else "A" for r in range(P)])
    slot = {"A": [7, 5, 3, 1], "B": [0, 2, 4, 6]}                     # plane rank in each raw window, left to right
    W = gap + 4 * (panel + gap); H = panel + 2 * top
    wins = {(s, i): (top, gap + i * (panel + gap)) for s in "AB" for i in range(4)}
    win_of = {r: (s, i) for s in "AB" for i, r in enumerate(slot[s])}
    aff = []                                                          # ref -> panel, reference plane P5 (rank 4)
    for r in range(P):
        if r == 4:
            aff.append(np.array([[1.0, 0, 0], [0, 1.0, 0]])); continue
        sc, th = 1 + rng.normal(0, 0.005, 2), np.radians(rng.normal(0, 0.3))
        A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * sc[:, None]
        aff.append(np.c_[A, rng.uniform(-3, 3, 2) + (np.eye(2) - A) @ [panel / 2, panel / 2]])
    pts = []
    while len(pts) < n_beads:
        for _ in range(5000):
            c = rng.uniform(32, panel - 33, 2)
            if all(np.hypot(*(c - q)) > min_sep for q in pts):
                pts.append(c); break
        else:
            break
    ref = np.array(pts); zb = rng.uniform(1.8, 2.6, len(ref)); E = peak * 2 * np.pi * sigma0 ** 2 * rng.uniform(0.7, 1.3, len(ref))
    z = step * np.arange(nz)
    clean = {s: np.full((nz, H, W), float(offset)) for s in "AB"}
    for s in "AB":
        for (s_, i), (t, l) in wins.items():
            if s_ == s:
                clean[s][:, t:t + panel, l:l + panel] += bg
    raw_pos = np.zeros((len(ref), P, 2))
    hw = 45
    for r in range(P):
        s, i = win_of[r]; t, l = wins[(s, i)]
        pp = R.apply_affine(aff[r], ref)
        xr = l + (panel - 1 - pp[:, 0] if s == "B" else pp[:, 0]); yr = t + pp[:, 1]
        raw_pos[:, r] = np.c_[xr, yr]
        zs = z + (step * b_frame_delay if s == "B" else 0)
        for b in range(len(ref)):
            x0, y0 = int(round(xr[b])), int(round(yr[b]))
            yy, xx = np.mgrid[y0 - hw:y0 + hw + 1, x0 - hw:x0 + hw + 1]
            m = (yy >= t) & (yy < t + panel) & (xx >= l) & (xx < l + panel)
            r2 = (xx - xr[b]) ** 2 + (yy - yr[b]) ** 2
            for k in range(nz):
                s2 = sigma0 ** 2 + (defocus_px_per_um * (zs[k] - zb[b] - delta[r])) ** 2
                img = E[b] * trans[r] * np.exp(-bleach * k) * np.exp(-r2 / (2 * s2)) / (2 * np.pi * s2)
                clean[s][k][yy[m], xx[m]] += img[m]
    streams = {s: (rng.poisson(np.clip(clean[s] - offset, 0, None)) + offset).astype(np.uint16) for s in "AB"}
    truth = dict(delta=delta, trans=trans, stream=stream, aff=aff, ref=ref, raw_pos=raw_pos, zb=zb, panel=panel,
                 windows={f"{s}{i}": (t, l, panel, panel) for (s, i), (t, l) in wins.items()}, win_of=win_of)
    return streams, z, truth


def write_dat(path, streams, z_start_nm, step_nm, exposure=0.05):
    """Interleaved big-endian .dat (frame 2k = A, 2k+1 = B) + .info JSON like the LabVIEW acquisitions."""
    T, H, W = streams["A"].shape
    inter = np.empty((2 * T, H, W), ">u2"); inter[0::2] = streams["A"]; inter[1::2] = streams["B"]
    inter.tofile(path)
    info = {"ROI CAM0": {"VWid": H, "HWid": W}, "ROI CAM1": {"VWid": H, "HWid": W}, "ExpoCam0": exposure, "ExpoCam1": exposure,
            "Seq Params": {"SeqType": "ZSTACK", "ZstackParams": {"StartZ": z_start_nm, "EndZ": z_start_nm + step_nm * T, "NZ": T}}}
    path.with_suffix(".info").write_text(json.dumps(info))


PRM = dict(level=0.5, smooth_px=3, percentiles=(2, 98), min_lit_px=20, gap_search=0.2, min_gap_depth=0.15, margin_px=10,
           smooth_z_um=0.2, smooth_xy_um=0.111, dog_um=(0.055, 0.44), tukey=0.25, candidates=("none", "lr", "ud", "lrud"),
           detect_sigma=6.0, peak_um=0.5, roi_half_um=0.67, core_half_um=0.111, isolation_um=1.55, radii_um=(0.44, 0.33, 0.22, 0.22),
           min_panels=6, focus_half_um=1.2, width_bounds_um=(0.05, 2.0), sigma0_um=0.4, max_shift_steps=2, edge_frames=2,
           min_snr=5.0, n_iter=30, clip_sigma=4.0, n_boot=50, lateral_um=1.0, gauss_sigma0_um=0.17, lm_iter=40, min_frames=4,
           affine_clip=3.0, affine_iter=5, trans_radius_um=2.0, trans_bg_um=0.5, trans_half_um=0.2)


def calibrate(streams, z, prm=PRM, pixel_um=PX, offset=100):
    """The notebook-01 chain on two camera streams (same steps, same order)."""
    p = prm; u = lambda um, **k: R.px(um, pixel_um, **k)
    wins = {s: R.detect_panels(R.mean_image(streams[s]), 4, p["level"], p["smooth_px"], p["percentiles"], p["min_lit_px"],
                               p["gap_search"], p["min_gap_depth"])[0] for s in "AB"}
    panels = R.common_windows(wins, p["margin_px"])
    ids = [q["id"] for q in panels]
    step = float(np.median(np.diff(z)))
    sm = (p["smooth_z_um"] / step, p["smooth_xy_um"] / pixel_um, p["smooth_xy_um"] / pixel_um)
    dogpx = tuple(d / pixel_um for d in p["dog_um"])
    raw = {q["id"]: io.to_signal(geometry.panel(streams, {**q, "orientation": "none"}), offset) for q in panels}
    mip_raw = {k: R.bead_projection(v, sm, dogpx) for k, v in raw.items()}
    scores = R.orientation_scores(mip_raw, "A0", p["candidates"], p["tukey"])
    orient_B = R.choose_orientation(scores, "B")[0]
    for q in panels:
        q["orientation"] = "none" if q["stream"] == "A" else orient_B
    stacks = [geometry.ORIENT[q["orientation"]](raw[q["id"]]) for q in panels]
    mips = {q["id"]: geometry.ORIENT[q["orientation"]](mip_raw[q["id"]]) for q in panels}
    shifts = {k: R.phase_correlation(mips["A0"], mips[k], p["tukey"])[0] for k in ids}
    roi = u(p["roi_half_um"])
    det = {k: R.detect_beads(mips[k], p["detect_sigma"], u(p["peak_um"], odd=True), roi + 3) for k in ids}
    pos, _, _ = R.match_beads(det, "A0", shifts, u(p["isolation_um"]), [u(r) for r in p["radii_um"]], p["min_panels"])
    zf = np.full(pos.shape[:2], np.nan)
    for j, q in enumerate(panels):
        ok = np.flatnonzero(np.isfinite(pos[:, j, 0]))
        curves = R.focus_signal(stacks[j], pos[ok, j], roi, u(p["core_half_um"]))
        for b, c in zip(ok, curves):
            z0, _, snr = R.fit_focus(z, c, p["focus_half_um"], p["width_bounds_um"], p["sigma0_um"], p["max_shift_steps"], p["edge_frames"])
            zf[b, j] = z0 if snr >= p["min_snr"] else np.nan
    delta, zb, zfc = R.plane_offsets(zf, p["n_iter"], p["clip_sigma"])
    order = np.argsort(delta)
    pf = np.full_like(pos, np.nan); sd = np.full_like(pos, np.nan)
    for j in range(len(panels)):
        pf[:, j], sd[:, j], _, _ = R.positions_at_focus(stacks[j], z, zfc[:, j], pos[:, j], p["lateral_um"], roi,
                                                        p["gauss_sigma0_um"] / pixel_um, p["lm_iter"], p["min_frames"])
    ref_j = int(order[4])
    aff, stats = R.register_planes(pf, ref_j, p["affine_clip"], p["affine_iter"], sd)
    kc = np.rint((np.nanmean(zfc, 1) - z[0]) / step).astype(int)
    T, Tsd, _, used = R.channel_transmission(stacks, pf, kc, int(round(p["trans_half_um"] / step)), u(p["trans_radius_um"]),
                                             u(p["trans_bg_um"]), p["n_boot"], 0)
    return dict(panels=panels, orient_B=orient_B, pos=pf, delta=delta, order=order, zf=zfc, aff=aff, stats=stats, ref_j=ref_j,
                T=T, Tsd=Tsd, n_trans=len(used), step=step)


def _to_panel(raw_xy, q):
    """Raw camera coordinates -> oriented-panel coordinates of a detected window."""
    w = q["window_raw"]; x = raw_xy[:, 0] - w["left"]; y = raw_xy[:, 1] - w["top"]
    if "lr" in q["orientation"]:
        x = w["width"] - 1 - x
    if "ud" in q["orientation"]:
        y = w["height"] - 1 - y
    return np.c_[x, y]


@pytest.fixture(scope="module")
def sim():
    streams, z, truth = simulate_bead_scan()
    return streams, z, truth, calibrate(streams, z)


def test_panels_and_orientation(sim):
    _, _, truth, out = sim
    assert out["orient_B"] == "lr"
    for q in out["panels"]:
        t, l, h, w = truth["windows"][q["id"]]
        wr = q["window_raw"]                                    # common crop centred in the true window
        assert abs(wr["top"] + wr["height"] / 2 - (t + h / 2)) <= 1.5 and abs(wr["left"] + wr["width"] / 2 - (l + w / 2)) <= 1.5
        assert 0 < h - wr["height"] <= 2 * 10 + 6


def test_plane_offsets_and_spacing(sim):
    _, _, truth, out = sim
    panel_rank = {f"{s}{i}": r for r, (s, i) in truth["win_of"].items()}
    ranks = np.array([panel_rank[q["id"]] for q in out["panels"]])
    assert np.array_equal(ranks[out["order"]], np.arange(8))                    # plane order recovered
    err = out["delta"] - truth["delta"][ranks]
    assert np.abs(err).max() < 0.015                                            # Δ within 15 nm
    sp = np.diff(np.sort(out["delta"]))
    assert abs(sp.mean() - np.diff(truth["delta"]).mean()) < 0.003
    per_bead = R.per_bead_spacing(out["zf"], out["order"])
    assert len(per_bead) >= 10 and abs(per_bead.mean() - np.diff(truth["delta"]).mean()) < 0.01


def test_affine_registration_recovers_true_positions(sim):
    _, _, truth, out = sim
    panel_rank = {f"{s}{i}": r for r, (s, i) in truth["win_of"].items()}
    q_of_rank = {panel_rank[q["id"]]: (j, q) for j, q in enumerate(out["panels"])}
    jr, qr = q_of_rank[4]
    assert jr == out["ref_j"]
    ref_true = _to_panel(truth["raw_pos"][:, 4], qr)
    for r in range(8):
        j, q = q_of_rank[r]
        pred = R.apply_affine(out["aff"][j], _to_panel(truth["raw_pos"][:, r], q))   # true panel positions -> reference
        assert np.sqrt(((pred - ref_true) ** 2).sum(1).mean()) < 0.05                    # px (≈ 5 nm)
        if r != 4:
            assert out["stats"][j]["loo_rms_px"] < 0.1


def test_channel_transmission(sim):
    _, _, truth, out = sim
    panel_rank = {f"{s}{i}": r for r, (s, i) in truth["win_of"].items()}
    ranks = np.array([panel_rank[q["id"]] for q in out["panels"]])
    assert out["n_trans"] >= 10
    assert np.abs(out["T"] - truth["trans"][ranks]).max() < 0.01                       # despite 1 %/frame bleaching
    assert np.all(out["Tsd"] < 0.01)


def test_camera_comb_detects_pairing_error():
    streams, z, truth = simulate_bead_scan(b_frame_delay=1, n_beads=25, seed=1)
    out = calibrate(streams, z)
    st = np.array([q["stream"] for q in out["panels"]])
    comb, d_unit = R.camera_comb(out["delta"], st, out["step"], assumed_shift=0)
    assert abs(d_unit - 0.35) < 0.02
    assert abs(abs(comb[1]["offset_delta"]) - 1) < 0.1            # the true pairing (B one step later) gives the designed comb
    assert abs(abs(comb[0]["offset_delta"]) - 1) > 0.4            # the assumed pairing is visibly off


def test_robust_fit_clips_at_declared_sigma():
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 100, (400, 2)); M = np.array([[1.01, 0.02, 3], [-0.01, 0.99, -2]])
    dst = R.apply_affine(M, src) + rng.normal(0, 0.1, src.shape)
    dst[:5] += 5                                                                        # outliers
    coef, keep, r, _ = R.robust_fit(src, dst, 1, 3.0, 10)
    assert not keep[:5].any() and keep[5:].mean() > 0.97                                # 3σ radial clip keeps ~98.9 % of inliers
    assert np.allclose(R.affine_from_coef(coef), M, atol=0.03)
    assert np.allclose(R.apply_affine(R.invert_affine(M), R.apply_affine(M, src)), src)


def test_design_spacing_and_axial_scale():
    inst = Instrument()
    M_l = inst.value("cameras.pixel_pitch_um") / inst.pixel_um("60x_water")
    dz = R.design_plane_spacing_um(inst.value("detection.prism_d_mm"), inst.value("detection.prism_n"), M_l)
    assert abs(M_l - 58.3) < 0.1 and abs(dz - 0.335) < 0.002                    # SI §6.2 formula, close to 347 ± 11 nm
    assert R.axial_scale(1.33, 1.33) == 1 and abs(R.axial_scale(1.33, 1.518) - 0.876) < 1e-3


def test_dat_roundtrip_of_simulation(tmp_path):
    streams, z, _ = simulate_bead_scan(panel=120, nz=6, n_beads=3, min_sep=20)
    f = tmp_path / "beads.dat"; write_dat(f, streams, 0, 200)
    acq = io.load(f, Instrument())
    assert np.array_equal(acq.streams["B"][3], streams["B"][3]) and np.allclose(acq.z_um["A"], z)
