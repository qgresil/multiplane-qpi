"""Tests of the uiPSF extension (notebook 02) on synthetic data; no uiPSF fit is run here."""
import json
from types import SimpleNamespace
import numpy as np
import pytest

from mpqpi import psf_uipsf as pu
from mpqpi.params import REG

OPTICS = dict(na=1.2, lam_um=0.515, n_imm=1.33, n_med=1.33, n_cov=1.52, px_um=0.111, dz_um=0.2, bead_radius_um=0.0875,
              with_apoid=True, stage_mov_dir="normal")


@pytest.fixture(autouse=True)
def clean_registry():
    REG.clear()
    yield


def _fake_acq(T=6, H=20, W=30, offset=100):
    t, y, x = np.mgrid[:T, :H, :W]
    A = (offset + 1000 * t + 30 * y + x).astype(np.uint16)
    B = (A + 500).astype(np.uint16)
    z = 1.0 - 0.2 * np.arange(T)                                      # decreasing commanded z
    return SimpleNamespace(streams={"A": A, "B": B}, z_um={"A": z, "B": z.copy()}), A, B


def test_export_geometry_orientation_offset_and_z_order(tmp_path):
    import tifffile
    acq, A, B = _fake_acq()
    calib = {"pixel_um": 0.111, "planes": [
        {"plane": "P1", "stream": "A", "panel_id": 0, "window_raw": {"top": 2, "left": 3, "height": 10, "width": 12},
         "orientation": "lr", "delta_z_um": -0.3, "affine_from_reference": [[1.02, 0, 1], [0, 1.02, 0]]},
        {"plane": "P2", "stream": "B", "panel_id": 1, "window_raw": {"top": 5, "left": 10, "height": 10, "width": 12},
         "orientation": "ud", "delta_z_um": 0.3, "affine_from_reference": [[1, 0, 0], [0, 1, 0]]}]}
    man = pu.export_planes(acq, calib, tmp_path, offset_counts=100)
    p1 = tifffile.imread(tmp_path / "P1.tif"); p2 = tifffile.imread(tmp_path / "P2.tif")
    assert p1.dtype == np.float32 and p1.shape == (6, 10, 12)
    exp1 = (A[::-1, 2:12, 3:15][..., ::-1]).astype(np.float32) - 100       # z ascending, lr flip, offset removed
    exp2 = (B[::-1, 5:15, 10:22][..., ::-1, :]).astype(np.float32) - 100
    assert np.array_equal(p1, exp1) and np.array_equal(p2, exp2)
    m1 = man["planes"][0]
    assert np.allclose(m1["z_um"], np.sort(acq.z_um["A"])) and abs(m1["step_um"] - 0.2) < 1e-12
    assert abs(m1["pixel_um"] - 0.111 / 1.02) < 1e-12 and man["planes"][1]["pixel_um"] == 0.111
    assert json.loads((tmp_path / "export.json").read_text())["offset_counts"] == 100


def test_effective_na_perfect_pupil_is_nominal():
    for n in (32, 64, 65):
        r, _ = pu.pupil_grid(n)
        assert abs(pu.effective_na(np.ones((n, n)), 1.2) - 1.2) < 1e-12
        assert abs(pu.effective_na(3.7 * np.ones((n, n)), 1.2) - 1.2) < 1e-12      # scale-free
    r, _ = pu.pupil_grid(128)
    assert abs(pu.effective_na((r < 0.75).astype(float), 1.2) - 0.9) < 0.01      # uniform disc of NA 0.9
    assert pu.effective_na(np.clip(1 - r, 0, None), 1.2) < 1.2                    # apodised pupil -> lower


def test_uipsf_params_contain_declared_values(tmp_path):
    prm = pu.uipsf_params(stack_file=tmp_path / "P3.tif", savename=tmp_path / "P3", pixel_um=0.108, step_um=0.2, na=1.2,
                          n_immersion=1.33, n_sample=1.0, n_coverslip=1.52, emission_um=0.515, bead_diameter_um=0.175,
                          gain_e_per_count=0.46, offset_counts=0.0, roi_xy_um=2.3, roi_z_um=4.2, detect_sigma_px=[2, 2, 2],
                          detect_kernel_px=[3, 3, 3], peak_height=0.2, max_beads=60, batch_size=100, var_photon=False,
                          var_blur=True, blur_sigma_px=0.5, const_pupilmag=False, symmetric_mag=True, with_apoid=True,
                          with_imm=False, n_max=8, pupil_size=64, iterations=150, stage_mov_dir="reverse")
    im = prm["option"]["imaging"]; md = prm["option"]["model"]
    assert im["NA"] == 1.2 and im["RI"] == {"imm": 1.33, "med": 1.0, "cov": 1.52} and im["emission_wavelength"] == 0.515
    assert prm["pixel_size"] == {"x": 0.108, "y": 0.108, "z": 0.2}
    assert prm["roi"]["roi_size"] == [21, 21, 21] and prm["roi"]["bead_radius"] == 0.0875     # 2.3/0.108 = 21.3 -> 21
    assert prm["gain"] == 0.46 and prm["ccd_offset"] == 0.0 and prm["stage_mov_dir"] == "reverse"
    assert md["var_photon"] is False and md["symmetric_mag"] is True and md["n_max"] == 8 and md["bin"] == 1
    assert prm["iteration"] == 150 and prm["PSFtype"] == "zernike_vector" and prm["swapxy"] is False
    assert pu.odd_px(2.0, 0.111) % 2 == 1 and pu.stage_mov_dir(True) == "normal" and pu.stage_mov_dir(False) == "reverse"
    json.dumps(prm)                                                                  # must be serialisable for the subprocess


def test_strict_overlay_rejects_unknown_keys():
    base = {"a": 1, "option": {"model": {"n_max": 8}}}
    pu._set_strict(base, {"option": {"model": {"n_max": 6}}})
    assert base["option"]["model"]["n_max"] == 6
    with pytest.raises(KeyError):
        pu._set_strict(base, {"option": {"model": {"nmax": 6}}})


def _zern_grid(n, n_max=4):
    """Noll Zernikes as in uiPSF genZern1 (small set, for the fake result)."""
    from math import factorial
    r, phi = pu.pupil_grid(n)
    def noll2nl(j):
        nn = int(np.ceil((-3 + np.sqrt(1 + 8 * j)) / 2)); l = j - nn * (nn + 1) // 2 - 1
        if nn % 2 != l % 2: l += 1
        return nn, (-l if j % 2 else l)
    Z = []
    for j in range(1, (n_max + 1) * (n_max + 2) // 2 + 1):
        nn, l = noll2nl(j); m = abs(l); g = np.sqrt(nn + 1) if m == 0 else np.sqrt(2 * nn + 2)
        R = sum(g * (-1) ** k * factorial(nn - k) / factorial(k) / factorial((nn + m) // 2 - k) / factorial((nn - m) // 2 - k)
                * r ** (nn - 2 * k) for k in range((nn - m) // 2 + 1))
        Z.append(R * (np.sin(m * phi) if l < 0 else np.cos(m * phi)))
    return np.array(Z)


def _write_fake_h5(path, astig_rad=0.4, sigma_yx=(0.7, 0.5), N=32, Lr=15, nz=11):
    import h5py
    fm = pu.ForwardModel(OPTICS, N, Lr)
    Zk = _zern_grid(N)
    zc = np.zeros((2, len(Zk))); zc[0, 0] = 1.0; zc[1, 5] = astig_rad                        # Noll 6 = astig 0°
    mag = np.maximum(np.tensordot(zc[0], Zk, 1), 0); ph = np.tensordot(zc[1], Zk, 1)
    pupil = fm.pupil(mag, ph)
    nzb = nz + 4
    Ib = fm.psf(pupil, (np.arange(nzb) - (nzb - 1) / 2) * OPTICS["dz_um"], sigma_yx, bead=True)
    nb = 3; photons, bg = 5000.0, 10.0
    fit = np.stack([photons * Ib[2:-2] + bg] * nb)
    data = np.random.default_rng(0).poisson(fit).astype(np.float32)
    prm = {"option": {"imaging": {"NA": 1.2, "emission_wavelength": 0.515, "RI": {"imm": 1.33, "med": 1.33, "cov": 1.52}},
                      "model": {"with_apoid": True}},
           "pixel_size": {"x": 0.111, "y": 0.111, "z": 0.2}, "roi": {"bead_radius": 0.0875}, "stage_mov_dir": "normal"}
    with h5py.File(path, "w") as f:
        f.attrs["params"] = json.dumps(prm)
        r = f.create_group("res")
        for k, v in dict(bg=np.full(nb, bg), intensity=np.full(nb, photons), pos=np.zeros((nb, 3)), zernike_coeff=zc,
                         zernike_polynomial=Zk, pupil=pupil.astype(np.complex64), sigma=np.array(sigma_yx),
                         apodization=fm.apod.astype(np.complex64), I_model_bead=Ib, I_model=Ib).items():
            r[k] = v
        g = f.create_group("rois")
        g["psf_data"] = data; g["psf_fit"] = fit; g["cor"] = np.array([[nz // 2, 20, 20]] * nb); g["image_size"] = [1, nz, 40, 40]
    return fm


def test_read_fake_result_and_conversions(tmp_path):
    h5 = tmp_path / "P1_zernike_vector_single.h5"
    _write_fake_h5(h5)
    res = pu.load_result(h5)
    assert res["intensity"].shape == res["data"].shape[:2] and res["optics"]["n_med"] == 1.33
    mag, ph, dev = pu.pupil_terms(res)
    assert dev < 1e-5 and abs(pu.effective_na(mag, res["optics"]["na"]) - 1.2) < 1e-9
    c, W, S = pu.phase_nm(res)
    assert abs(c[5] - 0.4 * 515 / (2 * np.pi)) < 1e-3 and abs(W - c[5]) < 1e-3 and 0 < S < 1
    b = pu.blur_sigma(res)
    assert b["sigma_yx_px"] == [0.7, 0.5] and abs(b["sigma_yx_nm"][1] - 55.5) < 1e-6
    fm = pu.ForwardModel(res["optics"], res["zernike_polynomial"].shape[-1], res["data"].shape[-1])
    assert pu.model_check(res, fm) < 1e-6
    rr, floor = pu.residuals(res)
    assert np.allclose(rr, floor, rtol=0.2)                                  # Poisson data around the model: residual ≈ noise floor


def test_budget_perfect_pupil_has_no_magnitude_or_phase_share(tmp_path):
    h5 = tmp_path / "x_zernike_vector_single.h5"
    _write_fake_h5(h5, astig_rad=0.0)
    res = pu.load_result(h5)
    fm = pu.ForwardModel(res["optics"], 32, res["data"].shape[-1])
    b = pu.budget(res, fm, (np.arange(41) - 20) * 0.1, oversample=3, axial_window_px=3)
    s = b["states"]
    assert abs(s["100"]["fwhm_lat_um"] - s["000"]["fwhm_lat_um"]) < 1e-6 and abs(s["010"]["fwhm_z_um"] - s["000"]["fwhm_z_um"]) < 1e-6
    assert s["111"]["fwhm_lat_um"] > s["000"]["fwhm_lat_um"]                 # σ 0.6 px blur wider than pixel integration
    assert np.allclose(b["share_range"]["blur"]["lateral"], 1.0, atol=1e-6)


def test_split_planes_recovers_trend_and_camera():
    dz = np.linspace(-1.2, 1.2, 8); cam = np.array(list("ABABABAB"))
    common = np.array([10.0, -5.0, 3.0]); trend = np.array([4.0, 0.0, 0.0]); camoff = np.array([0.0, 6.0, 0.0])
    coef = common + dz[:, None] * trend + np.where(cam == "A", 0.5, -0.5)[:, None] * camoff
    s = pu.split_planes(coef, dz, cam)
    assert np.allclose(s["common_nm"], common) and abs(s["W_camera"] - 6.0) < 1e-9
    assert abs(s["W_depth_trend"] - 4.0 * 1.2) < 1e-9 and s["W_rest"] < 1e-9
