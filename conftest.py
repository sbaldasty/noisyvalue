import importlib.util

# `--doctest-modules` imports every module under src/; skip the ones whose
# optional backend is not installed (their tests skip via importorskip).
collect_ignore = []
if importlib.util.find_spec("duet") is None:
    collect_ignore.append("src/noisyvalue/dduo.py")
