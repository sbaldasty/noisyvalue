import importlib.util

# `--doctest-modules` imports every module under src/; skip the ones whose
# optional backend is not installed (their tests skip via importorskip).


def _has_dduo_releases():
    """DDuo is importable and its mechanisms support `details=True`."""
    if importlib.util.find_spec("duet") is None:
        return False
    try:
        from duet.duet_vals import Release  # noqa: F401
    except ImportError:
        return False
    return True


collect_ignore = []
if not _has_dduo_releases():
    collect_ignore.append("src/noisyvalue/dduo.py")
