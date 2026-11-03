"""Core tests on simulated data: loaders, warp, and the SI model (a simulated bead must come back with the paper's numbers)."""
import json
import numpy as np
import pytest
from scipy import fft as sfft

from mpqpi import qpi, geometry, io
from mpqpi.config import Instrument
from mpqpi.params import REG


@pytest.fixture(autouse=True)
def clean_registry():
    REG.clear()
    yield


def test_registry_requires_tag_and_source():
    with pytest.raises(ValueError):
        REG.declare("x", 1, "guess", "because")
    with pytest.raises(ValueError):
        REG.declare("x", 1, "heuristic", "")
    assert REG.declare("x", 2, "heuristic", "test") == 2 and REG.heuristics()[0].name == "x"


def test_instrument_pixel_matches_si():
    px = Instrument().pixel_um("60x_water")
    assert abs(px - 0.1114) < 1e-3                       # SI §6.1: ≈ 111 nm


def test_dat_loader_interleaving_and_z(tmp_path):
    H, W, nz = 8, 6, 5
    frames = np.arange(2 * nz * H * W, dtype=">u2").reshape(2 * nz, H, W)
    f = tmp_path / "x.dat"; frames.tofile(f)
    info = {"ROI CAM0": {"VWid": H, "HWid": W}, "ExpoCam0": 0.05,
            "Seq Params": {"SeqType": "ZSTACK", "ZstackParams": {"StartZ": 0, "EndZ": 1000, "NZ": nz}}}
    f.with_suffix(".info").write_text(json.dumps(info))
    acq = io.load(f, Instrument())
    assert np.array_equal(acq.streams["A"][1], frames[2]) and np.array_equal(acq.streams["B"][1], frames[3])
    assert np.allclose(acq.z_um["A"], [0, 0.2, 0.4, 0.6, 0.8])          # step = (End − Start) / NZ by default


def test_warp_identity_and_shift():
    img = np.random.default_rng(0).random((20, 30)).astype(np.float32)
    assert np.allclose(geometry.warp(img, [[1, 0, 0], [0, 1, 0]]), img)
    sh = geometry.warp(img, [[1, 0, 2], [0, 1, 0]])                      # ref x -> panel x + 2
    assert np.allclose(sh[:, :-2], img[:, 2:])


def _simulated_point(na_det=1.15, na_ill=0.26, nz=50, ns=64, dz=0.2, px=0.111, sgn=+1):
    band = (0.5445, 0.6195)
    tf = qpi.transfer_function(na_det, na_ill, 1.33, band, n_samples=1_000_000)
    nzl = 160
    gz, gr = qpi.grids(nzl, ns, ns, dz, px)
    H = np.stack([qpi.lookup(tf, -sgn * g, gr)[0] for g in gz])
    gam = sfft.ifftn(1j * H); gam *= 1e-3 / np.abs(gam).max()
    I = np.fft.fftshift(1 + 2 * gam.real).astype(np.float32)[nzl // 2 - nz // 2:nzl // 2 + nz // 2]
    gzc = qpi.gz_cutoff(na_ill, band, 1.33, "si_text")
    return qpi.phase(qpi.cross_spectral_density(I, dz, px, tf, sgn, gzc), 3.15), gzc


def test_gz_cutoff_matches_si_text():
    assert abs(qpi.gz_cutoff(0.26, (0.5445, 0.6195), 1.33, "si_text") - 0.40) < 0.01   # SI §2: ≈ 0.4 rad/µm


def test_point_scatterer_is_positive_and_paper_width():
    ph, _ = _simulated_point()
    k, y, x = np.unravel_index(np.argmax(ph), ph.shape)
    assert ph.max() > 0 and abs(ph.max()) > abs(ph.min())
    prof = ph[k, y, :]; half = prof.max() / 2
    width_nm = 111 * np.sum(prof > half)                                 # coarse FWHM in pixels
    assert 200 < width_nm < 400                                          # SI Fig. S3b theory: 0.30 µm


def test_bead_theory_value():
    assert abs(qpi.bead_phase_theory(0.2, 1.59, 1.33, 0.58, 0.6) - 0.338) < 0.002   # SI §4: 0.34 rad
