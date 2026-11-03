"""Per-plane PSF characterisation with uiPSF (Liu et al., Nat. Methods 2024; github.com/ries-lab/uiPSF). EXTENSION — not in the paper.

The SI has no measured PSF: SI §6.3 is a Zemax design analysis (diffraction limited, Strehl > 0.95); the SI uses beads only for
co-registration, transmission and plane spacing (§6.2, §6.4). This module adds a measured, per-plane vectorial PSF model:

  export   per-plane oriented panel stacks of a bead z-scan (camera offset subtracted, frames sorted by commanded z) -> float32 TIFFs
  run      one uiPSF fit per plane ("zernike_vector", single channel), in the uiPSF environment, as a subprocess
           (python -m mpqpi.psf_uipsf <spec.json>); every setting that matters is passed explicitly (`uipsf_params`)
  analyse  learned pupil magnitude and phase, blur, fit residual vs photon noise, widths through focus, effective NA,
           per-slice photons vs raw signal, and a numpy re-implementation of uiPSF's forward model (`ForwardModel`) for the
           width budget ideal -> + magnitude -> + phase -> + blur

uiPSF conventions (checked in psflearning/learning/psfs/PSFZernikeBased_file.py and PSFInterface_file.py):
  pupil = aperture · apodization · M · exp(iΦ),  apodization = √cosθ_imm / cosθ_med (with_apoid),  M = max(Σ c_mag,j Z_j, 0),
  Φ = Σ c_phase,j Z_j (rad, Noll order, Noll normalisation -> coefficient = RMS); the learned M is therefore *relative to the
  ideal aplanatic pupil* of the same (NA, n_imm, n_med). Defocus phase = −2π k_z,imm · z (with_IMM = False): exact for a scan that
  changes the immersion thickness (stage or objective scan); the bead-depth (index-mismatch) aberration is then a constant per
  bead and goes into the learned phase / magnitude. Saved blur `sigma` = real-space Gaussian σ in camera pixels, order (σ_y, σ_x)
  (internally σ·π, filter exp(−2σ²f²)). With bin = 1 the model is sampled at pixel centres (no pixel integration), so the
  learned blur also absorbs the pixel width. stage_mov_dir = "reverse" makes uiPSF flip the stack along z, which flips the sign of
  the even-m Zernike terms (astigmatism, spherical, ...), odd-m terms (coma, trefoil) are unchanged.

The runner part (`_run_spec`) executes in the uiPSF environment: keep module-level imports to the standard library and numpy.
"""
import itertools, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

UIPSF_RESULT_SUFFIX = "_zernike_vector_single.h5"          # uiPSF appends _<PSFtype>_<channeltype> to savename
ZNAME = {5: "astig 45°", 6: "astig 0°", 7: "coma y", 8: "coma x", 9: "trefoil y", 10: "trefoil x", 11: "spherical",
         12: "2nd astig 0°", 13: "2nd astig 45°", 14: "quadrafoil 0°", 15: "quadrafoil 45°", 16: "2nd coma x", 17: "2nd coma y",
         22: "2nd spherical", 37: "3rd spherical"}


# ============================================================================================ export (extension)

def stage_mov_dir(increasing_z_towards_objective):
    """uiPSF `stage_mov_dir` for stacks exported in increasing commanded z (extension).
    uiPSF "normal" = the stage moves towards the objective along the stack. This is an instrument fact that the data cannot
    tell (only the sign of the even-m Zernike terms depends on it; between-plane comparisons do not)."""
    return "normal" if increasing_z_towards_objective else "reverse"


def plane_geometry(calib, entry):
    """Per-plane pixel (µm) and rotation (deg) from the plane's affine_from_reference (SI §6.4 affine, calibration.json).
    A maps reference (x, y) -> panel (x, y): one panel pixel = pixel_um / √|det A| in the sample. Extension (the paper uses one
    pixel size); it matters here because px·NA/λ sets the pupil scale in uiPSF."""
    A = np.asarray(entry.get("affine_from_reference", [[1, 0, 0], [0, 1, 0]]), float)
    s = np.sqrt(abs(np.linalg.det(A[:, :2])))
    return dict(pixel_um=float(calib["pixel_um"]) / s, scale=float(s), rotation_deg=float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))))


def export_planes(acq, calib, out_dir, offset_counts, planes=None, motion_px=None):
    """Write one float32 TIFF per plane: the plane's panel cut with `geometry.panel` (window_raw + orientation, i.e. oriented
    like the reference plane), camera offset subtracted (`io.to_signal`), frames sorted by increasing commanded z.
    No resampling between planes (each plane is fitted in its own frame) and no motion correction, unless `motion_px`
    (T, 2) = per-frame sample displacement (x, y) in oriented panel pixels is given: then every frame is shifted back by a
    periodic Fourier shift (extension, off by default; one trajectory for all planes).
    Returns the manifest written to <out_dir>/export.json."""
    from . import geometry, io
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import tifffile
    entries = [p for p in calib["planes"] if planes is None or p["plane"] in planes]
    man = dict(planes=[], offset_counts=float(offset_counts), motion_corrected=motion_px is not None, z_order="increasing commanded z")
    for p in entries:
        z = acq.z_um[p["stream"]]
        if z is None:
            raise ValueError(f"{p['plane']}: stream {p['stream']} has no commanded z (not a z-stack)")
        n = min(len(z), len(acq.streams[p["stream"]]))
        order = np.argsort(np.asarray(z[:n]), kind="stable")
        st = io.to_signal(geometry.panel({p["stream"]: acq.streams[p["stream"]][:n]}, p), offset_counts)[order]
        if motion_px is not None:
            mp = np.asarray(motion_px, float)[:n][order]
            st = np.stack([geometry.fourier_shift(f, (-dy, -dx)) for f, (dx, dy) in zip(st, mp)]).astype(np.float32)
        f = out_dir / f"{p['plane']}.tif"
        tifffile.imwrite(f, np.ascontiguousarray(st, np.float32))
        zs = np.asarray(z[:n], float)[order]
        geo = plane_geometry(calib, p)
        man["planes"].append(dict(plane=p["plane"], stream=p["stream"], panel_id=p.get("panel_id"), file=f.name,
                                  orientation=p["orientation"], window_raw=p["window_raw"], delta_z_um=p.get("delta_z_um"),
                                  shape=list(st.shape), z_um=zs.tolist(), step_um=float(np.median(np.diff(zs))), **geo))
    (out_dir / "export.json").write_text(json.dumps(man, indent=1))
    return man


# ============================================================================================ run (extension)

def odd_px(size_um, step_um):
    """Physical size -> odd number of samples (uiPSF ROIs are centred on a pixel)."""
    n = int(round(size_um / step_um))
    return n + 1 - n % 2


def uipsf_params(*, stack_file, savename, pixel_um, step_um, na, n_immersion, n_sample, n_coverslip, emission_um,
                 bead_diameter_um, gain_e_per_count, offset_counts, roi_xy_um, roi_z_um, detect_sigma_px, detect_kernel_px,
                 peak_height, max_beads, batch_size, var_photon, var_blur, blur_sigma_px, const_pupilmag, symmetric_mag,
                 with_apoid, with_imm, n_max, pupil_size, iterations, stage_mov_dir):
    """The uiPSF parameter overrides for one plane (extension). Everything that changes the fit is an argument (declared in the
    notebook); everything else stays at uiPSF's config_base + psftype/zernike + channeltype/1ch defaults, which the runner
    records. Fixed here: full panel and all slices (FOV 0), no CUDA, no plots, swapxy False (panels are already oriented),
    PSFtype zernike_vector / single channel / bead data, no initial pupil file, no drift estimation (uiPSF default).
    n_sample is uiPSF's RI.med: the same value must be used in every interpretation (`analyse_plane` reads it back)."""
    rx, rz = odd_px(roi_xy_um, pixel_um), odd_px(roi_z_um, step_um)
    return {
        "datapath": str(Path(stack_file).parent), "filelist": [str(stack_file)], "format": ".tif", "savename": str(savename),
        "stage_mov_dir": stage_mov_dir, "gain": float(gain_e_per_count), "ccd_offset": float(offset_counts),
        "pixel_size": {"x": float(pixel_um), "y": float(pixel_um), "z": float(step_um)},
        "roi": {"roi_size": [rz, rx, rx], "gauss_sigma": list(detect_sigma_px), "max_kernel": list(detect_kernel_px),
                "peak_height": float(peak_height), "max_bead_number": int(max_beads), "bead_radius": float(bead_diameter_um) / 2},
        "FOV": {"y_center": 0, "x_center": 0, "radius": 0, "z_start": 0, "z_end": 0, "z_step": 1},
        "option": {"imaging": {"emission_wavelength": float(emission_um), "NA": float(na),
                               "RI": {"imm": float(n_immersion), "med": float(n_sample), "cov": float(n_coverslip)}},
                   "model": {"pupilsize": int(pupil_size), "n_max": int(n_max), "blur_sigma": float(blur_sigma_px),
                             "var_blur": bool(var_blur), "with_apoid": bool(with_apoid), "const_pupilmag": bool(const_pupilmag),
                             "symmetric_mag": bool(symmetric_mag), "with_IMM": bool(with_imm), "var_photon": bool(var_photon),
                             "init_pupil_file": "", "bin": 1}},
        "PSFtype": "zernike_vector", "channeltype": "single", "datatype": "bead",
        "usecuda": False, "plotall": False, "swapxy": False, "iteration": int(iterations), "batch_size": int(batch_size),
    }


def result_file(savename):
    return Path(str(savename) + UIPSF_RESULT_SUFFIX)


def run_plane(params, python, log_file=None, overwrite=False, progress_every_s=30.0):
    """Run one uiPSF fit in the uiPSF environment (`python` = config.local()["uipsf_python"]) as a subprocess.
    Skips the fit if the result exists and was made with identical parameters; raises if it exists with other parameters
    (unless overwrite). Streams uiPSF's progress (stage changes, and at most every `progress_every_s` s). Returns the .h5 path."""
    h5 = result_file(params["savename"]); spec = Path(str(params["savename"]) + "_spec.json")
    if h5.exists() and not overwrite:
        old = json.loads(spec.read_text()).get("uipsf") if spec.exists() else None
        if old == json.loads(json.dumps(params)):
            print(f"{h5.name}: already fitted with these parameters, skipped")
            return h5
        raise RuntimeError(f"{h5} exists with different parameters: change the savename (tag) or pass overwrite=True")
    if not python or not Path(python).exists():
        raise RuntimeError(f"uiPSF python not found ({python!r}); set uipsf_python in ~/.mpqpi.yaml")
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps({"uipsf": params}, indent=1))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]),
               TF_CPP_MIN_LOG_LEVEL="2", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    log = open(log_file or str(params["savename"]) + "_log.txt", "w")
    t0 = last = time.time(); stage = None
    proc = subprocess.Popen([str(python), "-u", "-m", "mpqpi.psf_uipsf", str(spec)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
    buf = ""
    while True:
        ch = proc.stdout.read(256)
        if not ch:
            break
        log.write(ch); buf += ch
        *segs, buf = buf.replace("\r", "\n").split("\n")
        for s in (s.strip() for s in segs):
            if not s:
                continue
            st = s.split(":")[0][:40]
            if st != stage or time.time() - last > progress_every_s or s.startswith(("RESULT", "Traceback", "Error")):
                print(f"  [{time.time() - t0:5.0f} s] {s[:150]}", flush=True); stage, last = st, time.time()
    proc.wait(); log.close()
    if proc.returncode or not h5.exists():
        raise RuntimeError(f"uiPSF failed (exit {proc.returncode}); see {log.name}")
    return h5


def _set_strict(base, over, path=""):
    """Overlay `over` onto the uiPSF OmegaConf config; every key must already exist (a typo would otherwise be ignored)."""
    for k, v in over.items():
        if k not in base:
            raise KeyError(f"uiPSF has no parameter {path}{k}")
        if isinstance(v, dict):
            _set_strict(base[k], v, f"{path}{k}.")
        else:
            base[k] = v


def _run_spec(spec_file):
    """Executed in the uiPSF environment: load uiPSF defaults, overlay the spec, fit, save (extension)."""
    import matplotlib; matplotlib.use("Agg")
    from psflearning.psflearninglib import psflearninglib
    from psflearning import io as uio
    spec = json.loads(Path(spec_file).read_text())
    L = psflearninglib()
    L.param = uio.param.combine("config_base", psftype="zernike", channeltype="1ch")
    _set_strict(L.param, spec["uipsf"])
    t0 = time.time()
    images = L.load_data(); L.getpsfclass(); dataobj = L.prep_data(images)
    psfobj, fitter = L.learn_psf(dataobj, time=0)
    res = L.save_result(psfobj, dataobj, fitter)
    print("RESULT", res, f"{time.time() - t0:.0f} s", flush=True)


def uipsf_version(python):
    """git commit of the uiPSF package used by `python` (for provenance), or None."""
    try:
        loc = subprocess.run([str(python), "-c", "import psflearning, os; print(os.path.dirname(psflearning.__file__))"],
                             capture_output=True, text=True, timeout=120).stdout.strip()
        return subprocess.run(["git", "-C", loc, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ============================================================================================ read & analyse (extension)

def load_result(h5file):
    """Read a uiPSF single-channel result (.h5) into plain numpy arrays + the uiPSF parameters actually used."""
    import h5py
    with h5py.File(h5file, "r") as f:
        prm = json.loads(f.attrs["params"])
        r = f["res"]
        out = {k: r[k][()] for k in ("bg", "intensity", "pos", "zernike_coeff", "zernike_polynomial", "pupil", "sigma",
                                     "apodization", "I_model_bead", "I_model") if k in r}
        out.update(data=f["rois/psf_data"][()], fit=f["rois/psf_fit"][()], cor=f["rois/cor"][()],
                   image_size=f["rois/image_size"][()], params=prm, file=str(h5file))
    nb, nz = out["data"].shape[:2]
    out["intensity"] = np.broadcast_to(np.reshape(out["intensity"], (nb, -1)), (nb, nz)).copy()   # var_photon False: one per bead
    out["bg"] = np.reshape(out["bg"], (nb,))
    out["zernike_coeff"] = np.reshape(out["zernike_coeff"], (2, -1))
    im = prm["option"]["imaging"]
    out["optics"] = dict(na=im["NA"], lam_um=im["emission_wavelength"], n_imm=im["RI"]["imm"], n_med=im["RI"]["med"],
                         n_cov=im["RI"]["cov"], px_um=prm["pixel_size"]["x"], dz_um=prm["pixel_size"]["z"],
                         bead_radius_um=prm["roi"]["bead_radius"], with_apoid=prm["option"]["model"]["with_apoid"],
                         stage_mov_dir=prm["stage_mov_dir"])
    return out


def pupil_grid(n):
    """uiPSF pupil grid: normalised radius r = NA_ray / NA and azimuth, aperture r < 1."""
    k = np.linspace(-1 + 1 / n, 1 - 1 / n, n)
    kx, ky = np.meshgrid(k, k)
    return np.hypot(kx, ky), np.arctan2(ky, kx)


def pupil_terms(res):
    """Learned magnitude M (relative to the ideal aplanatic pupil, i.e. 1 = perfect objective, arbitrary overall scale),
    phase Φ (rad) and the consistency of M·apod·e^{iΦ} with the saved pupil (relative max deviation)."""
    zc, Zk = res["zernike_coeff"], res["zernike_polynomial"]
    r, _ = pupil_grid(Zk.shape[-1]); ap = r < 1
    mag = np.maximum(np.tensordot(zc[0], Zk, 1), 0)
    phase = np.tensordot(zc[1], Zk, 1)
    rebuilt = ap * res["apodization"] * mag * np.exp(1j * phase)
    dev = float(np.abs(rebuilt - res["pupil"]).max() / np.abs(res["pupil"]).max())
    return mag, phase, dev


def effective_na(mag, na):
    """Effective NA of a pupil magnitude M given relative to the ideal aplanatic pupil (extension).
    NA_eff = NA · √(⟨r²⟩_w / ⟨r²⟩_ideal), weight w = M² over the aperture, ideal = uniform weight on the same grid,
    i.e. the NA of a uniformly filled pupil with the same light-weighted mean NA². A perfect pupil (M constant) returns NA."""
    r, _ = pupil_grid(mag.shape[-1]); ap = r < 1
    w = np.where(ap, mag, 0.0) ** 2
    return float(na * np.sqrt(((w * r ** 2).sum() / w.sum()) / (r[ap] ** 2).mean()))


def radial_profile(mag, na, n_bins, n_sectors, inner_frac):
    """Azimuthal mean of M vs NA, the min/max over `n_sectors` angular sectors, normalised to the mean of M inside
    r < inner_frac (so 1 = as transmissive as the pupil centre)."""
    r, phi = pupil_grid(mag.shape[-1]); ap = r < 1
    m = np.where(ap, mag, np.nan); m = m / np.nanmean(m[r < inner_frac])
    e = np.linspace(0, 1, n_bins + 1)
    sec = np.floor((phi + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    prof = np.array([np.nanmean(m[(r >= a) & (r < b)]) for a, b in zip(e[:-1], e[1:])])
    sp = np.array([[np.nanmean(m[(r >= a) & (r < b) & (sec == s)]) for a, b in zip(e[:-1], e[1:])] for s in range(n_sectors)])
    return na * (e[1:] + e[:-1]) / 2, prof, np.nanmin(sp, 0), np.nanmax(sp, 0), m


def phase_nm(res):
    """Zernike phase coefficients in nm RMS (Noll j = 1..J; coefficient (rad) · λ / 2π) and the wavefront RMS over j ≥ 5
    (piston, tilts and defocus = bead position) with its Maréchal Strehl exp(−(2πW/λ)²)."""
    lam_nm = 1e3 * res["optics"]["lam_um"]
    c = res["zernike_coeff"][1] * lam_nm / (2 * np.pi)
    W = float(np.sqrt((c[4:] ** 2).sum()))
    return c, W, float(np.exp(-(2 * np.pi * W / lam_nm) ** 2))


def blur_sigma(res):
    """Learned blur as saved by uiPSF: real-space σ in camera pixels, order (σ_y, σ_x), and in nm. The model has no pixel
    integration, so a perfect detector already needs ≈ 1/√12 px (Gaussian approximation of the box); `beyond_pixel_nm`
    = √(σ² − 1/12) px is therefore only indicative."""
    s = np.ravel(res["sigma"]).astype(float); px = 1e3 * res["optics"]["px_um"]
    return dict(sigma_yx_px=s.tolist(), sigma_yx_nm=(s * px).tolist(), pixel_alone_nm=px / np.sqrt(12),
                beyond_pixel_yx_nm=(np.sqrt(np.clip(s ** 2 - 1 / 12, 0, None)) * px).tolist())


def residuals(res):
    """Per-bead relative residual ‖d − m‖ / ‖d − bg‖ and the photon-noise floor √(Σ m) / ‖d − bg‖ (Poisson variance = model;
    camera read noise not included, so the floor is a lower bound)."""
    d, m, bg = res["data"], res["fit"], res["bg"][:, None, None, None]
    sig = np.sqrt(((d - bg) ** 2).sum((1, 2, 3)))
    return np.sqrt(((d - m) ** 2).sum((1, 2, 3))) / sig, np.sqrt(np.clip(m, 0, None).sum((1, 2, 3))) / sig


def bead_average(res):
    """Background-subtracted data and model, each bead normalised to unit total, averaged (identically for both)."""
    bg = res["bg"][:, None, None, None]
    w = 1 / (res["data"] - bg).sum((1, 2, 3))[:, None, None, None]
    return ((res["data"] - bg) * w).mean(0), ((res["fit"] - bg) * w).mean(0)


def gauss_fwhm(img):
    """FWHM (x, y) in pixels of an elliptical 2D Gaussian + offset fitted to one image (the image as recorded: pixel and bead
    included). NaN if the fit fails or ends on a width bound (far from focus the fit locks onto noise or a ring).
    Bounds follow the image size."""
    from scipy import optimize
    ny, nx = img.shape
    lo, hi = (0.2, 0.2), (nx / 2, ny / 2)
    yy, xx = np.mgrid[:ny, :nx].astype(float); yy -= ny // 2; xx -= nx // 2
    if not np.isfinite(img).all() or img.max() <= 0:
        return np.full(2, np.nan)
    g = lambda c, A, x0, y0, sx, sy, b: A * np.exp(-0.5 * ((c[0] - x0) ** 2 / sx ** 2 + (c[1] - y0) ** 2 / sy ** 2)) + b
    try:
        p, _ = optimize.curve_fit(g, (xx.ravel(), yy.ravel()), img.ravel(), p0=[img.max(), 0, 0, 1.5, 1.5, 0],
                                  bounds=([0, -nx / 4, -ny / 4, *lo, -np.inf], [np.inf, nx / 4, ny / 4, *hi, np.inf]),
                                  maxfev=4000)
        s = p[3:5].copy()
        s[(s < 1.01 * np.array(lo)) | (s > 0.99 * np.array(hi))] = np.nan
        return 2 * np.sqrt(2 * np.log(2)) * s
    except (RuntimeError, ValueError):
        return np.full(2, np.nan)


def axial_fwhm(prof, z):
    """Full width at half maximum of an axial profile (linear interpolation); NaN if it does not drop below half on both sides."""
    prof = np.asarray(prof, float); i0 = int(np.nanargmax(prof)); hm = prof[i0] / 2
    l = i0
    while l > 0 and prof[l] > hm: l -= 1
    r = i0
    while r < len(prof) - 1 and prof[r] > hm: r += 1
    if prof[l] > hm or prof[r] > hm:
        return np.nan
    return float(np.interp(hm, [prof[r], prof[r - 1]], [z[r], z[r - 1]]) - np.interp(hm, [prof[l], prof[l + 1]], [z[l], z[l + 1]]))


def on_axis(vol, window_px):
    """Mean of the central window_px × window_px pixels per slice (axial profile)."""
    ny, nx = vol.shape[-2:]; h = window_px // 2
    return vol[..., ny // 2 - h:ny // 2 + h + 1, nx // 2 - h:nx // 2 + h + 1].mean((-1, -2))


def widths_through_focus(res, axial_window_px):
    """Per bead and slice: lateral Gaussian FWHM (x, y) of data and model (px); per bead: focus slice and axial FWHM (µm) of the
    on-axis profile, data and model. Returns a dict."""
    bg = res["bg"][:, None, None]
    d, m = res["data"], res["fit"]; nb, nz = d.shape[:2]
    fd = np.array([[gauss_fwhm(d[b, k] - bg[b]) for k in range(nz)] for b in range(nb)])
    fm = np.array([[gauss_fwhm(m[b, k] - bg[b]) for k in range(nz)] for b in range(nb)])
    kb = np.argmax((d - bg[:, None]).max((2, 3)), 1)
    z = (np.arange(nz) - nz // 2) * res["optics"]["dz_um"]
    ad = [axial_fwhm(on_axis(d[b] - bg[b], axial_window_px), z) for b in range(nb)]
    am = [axial_fwhm(on_axis(m[b] - bg[b], axial_window_px), z) for b in range(nb)]
    foc = lambda a: np.array([a[b, kb[b]] for b in range(nb)])
    return dict(fwhm_data_px=fd, fwhm_model_px=fm, focus_slice=kb, z_um=z, axial_data_um=np.array(ad), axial_model_um=np.array(am),
                focus_data_px=np.nanmedian(foc(fd), 0), focus_model_px=np.nanmedian(foc(fm), 0))


def raw_signal_check(res, stack, gain, disc_um, ring_um):
    """Per-slice photons fitted by uiPSF vs the raw signal of the same beads (extension diagnostic).
    Raw = sum over a disc of radius disc_um around each bead minus the median of the ring disc_um..ring_um, × gain (photons),
    over the frames of each bead's ROI. If uiPSF's per-slice photons (var_photon) change much more than the raw signal, they are
    compensating for an axial shape the model gets wrong; with var_photon False they are constant and the raw curve tests
    that assumption (bleaching). Returns per-bead curves (normalised to their own focus value) and the absolute ratio at focus."""
    px = res["optics"]["px_um"]; nb, nz = res["data"].shape[:2]
    rd, rb = disc_um / px, ring_um / px; R = int(np.ceil(rb))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; rr = np.hypot(yy, xx)
    raw, fit, kfoc = [], [], []
    kb = np.argmax((res["data"] - res["bg"][:, None, None, None]).max((2, 3)), 1)
    for b in range(nb):
        zc, yc, xc = (int(round(v)) for v in res["cor"][b])
        z0 = zc - nz // 2
        if not (R <= yc < stack.shape[1] - R and R <= xc < stack.shape[2] - R and 0 <= z0 and z0 + nz <= stack.shape[0]):
            continue
        sub = stack[z0:z0 + nz, yc - R:yc + R + 1, xc - R:xc + R + 1]
        bgv = np.median(sub[:, (rr > rd) & (rr <= rb)], 1)
        raw.append(gain * (sub - bgv[:, None, None])[:, rr <= rd].sum(1)); fit.append(res["intensity"][b]); kfoc.append(kb[b])
    if not raw:
        return None
    raw, fit, kfoc = np.array(raw), np.array(fit), np.array(kfoc)
    at = lambda a: np.array([a[i, k] for i, k in enumerate(kfoc)])
    change = lambda a: float(np.median(a[:, -3:].mean(1) / a[:, :3].mean(1)))
    return dict(raw_photons=raw, fit_photons=fit, raw_norm=raw / at(raw)[:, None], fit_norm=fit / at(fit)[:, None],
                ratio_fit_over_raw_at_focus=float(np.median(at(fit) / at(raw))),
                change_fit=change(fit), change_raw=change(raw), n_beads=len(raw))


# ---------------------------------------------------------------- uiPSF forward model, numpy port (for the width budget)

def _czt_params(kpix, px, N, M):
    """uiPSF utilities.prechirpz1 (numpy): chirp-z from an N×N pupil grid (step kpix) to an M×M image grid (step px)."""
    k = np.linspace(-N / 2 + 0.5, N / 2 - 0.5, N); xk, yk = np.meshgrid(k, k)
    x = np.linspace(-M / 2 + 0.5, M / 2 - 0.5, M); xr, yr = np.meshgrid(x, x)
    b = np.linspace(-(N + M) / 2 + 1, (N + M) / 2 - 1, N + M - 1); xb, yb = np.meshgrid(b, b)
    a = 1j * np.pi * kpix * px
    return np.exp(a * (xk ** 2 + yk ** 2)), np.fft.fft2(np.exp(-a * (xb ** 2 + yb ** 2))), np.exp(a * (xr ** 2 + yr ** 2))


def _czt(data, prm):
    """uiPSF utilities.cztfunc1 (numpy), over the last two axes."""
    A, Bh, C = prm; N, Lb, M = A.shape[0], Bh.shape[0], C.shape[0]
    pad = np.zeros(data.shape[:-2] + (Lb, Lb), complex); pad[..., :N, :N] = A * data / N
    return C * np.fft.ifft2(np.fft.fft2(pad) * Bh / Lb)[..., -M:, -M:]


def _fft3c(a):
    ax = (-3, -2, -1); return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(a, axes=ax), axes=ax), axes=ax)


def _ifft3c(a):
    ax = (-3, -2, -1); return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(a, axes=ax), axes=ax), axes=ax)


class ForwardModel:
    """uiPSF's vectorial bead model (PSFInterface.calpupilfield + PSFZernikeBased.genpsfmodel + gen_bead_kernel), re-implemented
    in numpy so that terms can be switched on one by one (extension). Same pupil grid, dipole fields with Fresnel factors,
    apodization √cosθ_imm / cosθ_med, chirp-z transform, bead kernel j_{3/2}, Gaussian filter exp(−2σ²f²) (σ internal = π σ_px).
    `oversample` > 1 samples the image on a b× finer grid and sums b × b blocks (exact pixel integration, like uiPSF `bin`)."""

    def __init__(self, optics, n_pupil, roi_px):
        o = self.o = dict(optics); self.N, self.L = int(n_pupil), int(roi_px)
        sq = np.lib.scimath.sqrt
        r, phi = pupil_grid(self.N); self.r = r
        na, lam, ni, nm, nc = o["na"], o["lam_um"], o["n_imm"], o["n_med"], o["n_cov"]
        ci, cm, cc = sq(1 - (r * na / ni) ** 2), sq(1 - (r * na / nm) ** 2), sq(1 - (r * na / nc) ** 2)
        self.kz = sq((ni / lam) ** 2 - (r * na / lam) ** 2)                                   # 1/µm (immersion)
        Tp = (2 * nm * cm / (nm * cc + nc * cm)) * (2 * nc * cc / (nc * ci + ni * cc))
        Ts = (2 * nm * cm / (nm * cm + nc * cc)) * (2 * nc * cc / (nc * cc + ni * ci))
        pv = Tp * np.stack([cm * np.cos(phi), cm * np.sin(phi), -r * na / nm])
        sv = Ts * np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
        self.dipole = np.concatenate((np.cos(phi) * pv - np.sin(phi) * sv, np.sin(phi) * pv + np.cos(phi) * sv), 0)
        self.aperture = (r < 1).astype(float)
        self.apod = sq(ci) / cm if o.get("with_apoid", True) else np.ones_like(r)
        self._norm = {}

    def pupil(self, mag=None, phase=None):
        """aperture · apodization · M · e^{iΦ}; M = 1, Φ = 0 when omitted (the ideal aplanatic pupil)."""
        m = 1.0 if mag is None else mag; p = 0.0 if phase is None else phase
        return self.aperture * self.apod * m * np.exp(1j * p)

    def _intensity(self, pupil, z_um, b):
        M = self.L * b
        prm = _czt_params(2 * self.o["na"] / self.o["lam_um"] / self.N, self.o["px_um"] / b, self.N, M)
        if b not in self._norm:                                              # uiPSF normf: ideal in-focus pupil sums to 1
            p0 = self.pupil()
            self._norm[b] = 1 / sum((np.abs(_czt(p0 * h, prm)) ** 2).sum() for h in self.dipole)
        ph = np.exp(-2j * np.pi * self.kz[None] * np.asarray(z_um, float)[:, None, None])
        return sum(np.abs(_czt(pupil[None] * ph * h[None], prm)) ** 2 for h in self.dipole) * self._norm[b]

    def bead_kernel(self, nz, dz_um, b=1):
        """uiPSF gen_bead_kernel: FT of a uniform sphere of radius bead_radius_um, normalised to max 1."""
        from scipy import special
        R = self.o["bead_radius_um"]; M = self.L * b; px = self.o["px_um"] / b
        if R <= 0:
            return np.ones((nz, M, M))
        x = np.linspace(-M / 2 + 0.5, M / 2 - 0.5, M) + 1e-6; zr = np.linspace(-nz / 2 + 0.5, nz / 2 - 0.5, nz)
        q = np.sqrt((x[None, :, None] / (M * px)) ** 2 + (x[None, None, :] / (M * px)) ** 2 + (zr[:, None, None] / (nz * dz_um)) ** 2) * R
        k = special.jv(1.5, 2 * np.pi * q) / q ** 1.5 * R ** 3
        return k / k.max()

    def psf(self, pupil, z_um, sigma_yx_px=None, bead=True, oversample=1):
        """Image stack (nz, L, L) on the camera grid for slices at z_um (uniform grid). sigma_yx_px = learned blur (camera px)
        or None (no blur); oversample > 1 = pixel integration."""
        b = int(oversample); M = self.L * b; z = np.asarray(z_um, float)
        I = self._intensity(pupil, z, b)
        F = _fft3c(I)
        if sigma_yx_px is not None:
            s = np.asarray(sigma_yx_px, float) * np.pi * b                 # internal units on the (fine) grid
            k = np.linspace(-M / 2 + 0.5, M / 2 - 0.5, M) / M
            f2 = np.exp(-2 * s[1] ** 2 * k[None, :] ** 2 - 2 * s[0] ** 2 * k[:, None] ** 2); F = F * (f2 / f2.max())
        if bead:
            F = F * self.bead_kernel(len(z), z[1] - z[0] if len(z) > 1 else self.o["dz_um"], b)
        V = np.real(_ifft3c(F))
        return V.reshape(len(z), self.L, b, self.L, b).sum((2, 4)) if b > 1 else V


def model_check(res, fm):
    """Rebuilt forward model (saved pupil and blur, bead) vs uiPSF's own I_model_bead: max |difference| / max (both normalised
    per slice sum). Should be ≲ 1e-3 when the port is exact."""
    Ib = res["I_model_bead"]; nz = Ib.shape[0]
    z = (np.arange(nz) - (nz - 1) / 2) * res["optics"]["dz_um"]
    mine = fm.psf(res["pupil"], z, np.ravel(res["sigma"]), bead=True)
    nrm = lambda a: a / a.sum((1, 2), keepdims=True)
    return float(np.abs(nrm(mine) - nrm(Ib)).max() / nrm(Ib).max())


def budget(res, fm, z_um, oversample, axial_window_px):
    """Width budget (extension): forward model with each learned term on/off — magnitude M ("NA loss"), phase Φ
    ("aberrations"), blur (learned σ in place of pixel integration) — on a fine z grid; ideal = M 1, Φ 0, pixel integration
    (`oversample`), same bead and same (NA, n_imm, n_sample) as the fit. The share of each term depends on the order in which
    the terms are added (they do not add in quadrature exactly), so the shares are given for all 3! orders (min, max).
    Returns the per-state metrics (lateral FWHM x/y and axial FWHM, µm) and the shares."""
    mag, phase, _ = pupil_terms(res)
    mag = mag / np.mean(mag[fm.r < 1])
    px = res["optics"]["px_um"]; sig = np.ravel(res["sigma"])
    states, vols = {}, {}
    for on in itertools.product((0, 1), repeat=3):
        p = fm.pupil(mag if on[0] else None, phase if on[1] else None)
        V = fm.psf(p, z_um, sig if on[2] else None, bead=True, oversample=1 if on[2] else oversample)
        k = int(np.argmax(V.max((1, 2))))
        fx, fy = gauss_fwhm(V[k] / V[k].max()) * px
        a = on_axis(V, axial_window_px)
        states[on] = dict(fwhm_x_um=float(fx), fwhm_y_um=float(fy), fwhm_lat_um=float((fx + fy) / 2),
                          fwhm_z_um=axial_fwhm(a / a.max(), np.asarray(z_um)), focus=k)
        vols[on] = V
    names = ("magnitude", "phase", "blur")
    shares = {n: {"lateral": [], "axial": []} for n in names}
    for order in itertools.permutations(range(3)):
        on = [0, 0, 0]
        for t in order:
            before = tuple(on); on[t] = 1; after = tuple(on)
            for key, q in (("lateral", "fwhm_lat_um"), ("axial", "fwhm_z_um")):
                tot = states[(1, 1, 1)][q] ** 2 - states[(0, 0, 0)][q] ** 2
                shares[names[t]][key].append((states[after][q] ** 2 - states[before][q] ** 2) / tot if tot else np.nan)
    rng = {n: {k: [float(np.nanmin(v)), float(np.nanmax(v))] for k, v in d.items()} for n, d in shares.items()}
    seq = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]                       # the order shown: magnitude -> phase -> blur
    return dict(states={"".join(map(str, k)): v for k, v in states.items()}, sequence=["".join(map(str, s)) for s in seq],
                share_range=rng, volumes=vols)


def split_planes(coef, delta_z_um, streams):
    """Plane-to-plane decomposition of per-plane Zernike coefficients coef (P, J) in nm (extension; idea: SI §6.2–6.3, the prism
    paths differ per plane while everything before the split is shared): common = mean over planes; plane-specific = the rest,
    fitted as a linear trend with the plane's Δz + a camera (stream A − B) offset + residual. RMS values in nm."""
    coef = np.asarray(coef, float); P = len(coef)
    common = coef.mean(0); dev = coef - common
    out = dict(W_total=np.sqrt((coef ** 2).sum(1)).tolist(), W_common=float(np.sqrt((common ** 2).sum())),
               W_plane_specific=np.sqrt((dev ** 2).sum(1)).tolist(), common_nm=common.tolist(),
               W_depth_trend=np.nan, W_camera=np.nan, W_rest=np.nan)
    camA = np.array([s == "A" for s in streams]); dz = np.asarray(delta_z_um, float)
    if P >= 4 and camA.any() and (~camA).any() and np.isfinite(dz).all():
        X = np.c_[np.ones(P), dz - dz.mean(), np.where(camA, 0.5, -0.5)]
        bt = np.linalg.lstsq(X, coef, rcond=None)[0]
        out.update(W_depth_trend=float(np.sqrt((bt[1] ** 2).sum()) * (dz.max() - dz.min()) / 2),
                   W_camera=float(np.sqrt((bt[2] ** 2).sum())), W_rest=float(np.sqrt(np.mean(((coef - X @ bt) ** 2).sum(1)))))
    return out


if __name__ == "__main__":
    _run_spec(sys.argv[1])
