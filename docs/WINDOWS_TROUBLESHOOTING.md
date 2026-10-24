# ReliaGraph Windows startup troubleshooting

## If the launcher says `.venv` was not created or cannot be found

`.venv` is a private Python environment that the launcher creates automatically. It is not something you need to download separately. A failed creation usually means that the Python version/launcher was not found, the existing `.venv` folder is incomplete, the folder is not writable, or Python was installed without the required components.

1. Extract the entire ZIP to a normal writable folder such as `C:\ReliaGraph` or a folder under Documents. Do not run the `.bat` from inside the ZIP preview or from a temporary extraction folder.
2. Install 64-bit Python 3.14 (preferred), or 3.13, 3.12 or 3.11, from https://www.python.org/downloads/windows/. Enable **Add Python to PATH** during installation.
3. Close old terminals using this project, then double-click `run_reliagraph_windows.bat`. The launcher checks supported Python versions, creates `.venv`, and uses `.venv\Scripts\python.exe` directly (it does not depend on activation working).
4. If a previous `.venv` folder was incomplete, the launcher tries to rename it and create a fresh environment. If it cannot rename it, close VS Code and Python windows, delete the `.venv` folder inside the extracted project, then run the launcher again. Do not delete any folder named `data`; it may contain your local workspace.
5. The first launch needs internet access to install packages. If `pip` shows a network/DNS error, reconnect to the internet and run the same launcher again.

## If it still fails

Open Command Prompt in the extracted project folder (click File Explorer's address bar, type `cmd`, press Enter) and run:

```bat
py -0p
python --version
where python
```

Send the exact error text and the output of these commands. Do not send passwords or private data.

## Multilingual mode

First launch the baseline successfully with `run_reliagraph_windows.bat`. Only then run `run_reliagraph_multilingual_windows.bat`. It installs the extra embedding package and may download model weights on first use; successful installation alone does not prove Hindi-English retrieval quality.
