"""Per-session alpha facts read off the collection HDF5.

Both answers need the file open, so the suite records them into
results/<task>/alpha/sessions.json and the plotting side reads them from there
rather than reopening every session.
"""
import h5py


def has_blend_fields(full_file):
    """True if the run logs arbitrator components (i.e. a blended approach)."""
    try:
        with h5py.File(full_file, "r") as f:
            for demo in f["data"]:
                return "blend_similarity" in f["data"][demo]
    except Exception:
        pass
    return False


def alpha_is_constant(full_file, eps=1e-6):
    """True if the round's alphas have ~no variance (e.g. unlogged constant)."""
    try:
        with h5py.File(full_file, "r") as f:
            for demo in f["data"]:
                g = f["data"][demo]
                if "alphas" in g:
                    return float(g["alphas"][:].std()) < eps
    except Exception:
        pass
    return False
