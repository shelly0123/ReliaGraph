# Python 3.14 compatibility note

## Recommendation
ReliaGraph's core local dashboard is configured to prefer CPython 3.14 (64-bit), falling back to 3.13, 3.12 or 3.11 if 3.14 is not installed. The application code itself does not depend on a Python-3.14-only feature; compatibility is primarily controlled by the installed dependency wheels.

## Dependency basis checked on 2026-10-24
- Python.org lists Python 3.14.9 as a stable Windows release.
- NumPy 2.4/2.5 supports Python 3.14.
- SciPy 1.16.1 introduced stable Python 3.14 wheels.
- Current scikit-learn releases document support for Python 3.14.
- Current Streamlit and FastAPI documentation/package metadata include Python 3.14.
- PyTorch's current install guidance lists Python 3.10–3.14 on Windows; current Sentence Transformers metadata includes Python 3.14.

These package-level compatibility statements do not replace a complete installation test on the user's Windows machine. The optional GLiNER extraction stack and the exact transitive dependency resolution are not verified under Python 3.14 in this build environment.

## What was actually tested here
The available build runtime is Python 3.13.5. The project test suite and dashboard acceptance suite were not executed under Python 3.14 because that interpreter is not installed in the build environment and outbound package downloads are unavailable. Therefore this package is **configured for Python 3.14, but is not certified as fully tested on Python 3.14**.

## Windows usage
Install Python 3.14 (64-bit), then launch `run_reliagraph_windows.bat`. On a fresh extraction, the launcher selects Python 3.14 and creates `.venv`. If the folder already contains a valid `.venv` created with Python 3.11–3.13, the launcher reuses it. To recreate the environment with Python 3.14, close the app, rename or delete only the project's `.venv` folder, then rerun the launcher. Do not delete `data`, as it may contain the local workspace.
