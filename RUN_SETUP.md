# Run Button Setup

These changes let you start the dashboard from VS Code's Run button instead of typing `streamlit run app.py`.

## How to run

1. Open `app.py` in VS Code.
2. Click ▶ **Run Python File**, or press **F5**.
3. The dashboard opens in your browser at http://localhost:8501.

To stop it, press **Ctrl+C** in the terminal, or click the Stop button if you started it with F5.

`streamlit run app.py` from the terminal still works as before.

## What changed

### 1. `app.py`: starts itself under Streamlit

Streamlit apps must be started with `streamlit run`. Running them with plain `python` doesn't start the app, and that is what the ▶ button does. A short block near the top of `app.py` now checks how the file was started. If Streamlit isn't running it, the block relaunches the file with `streamlit run` and opens the browser:

```python
if __name__ == "__main__":
    # Started with plain `python app.py` (e.g. VS Code's Run button): relaunch under Streamlit.
    from streamlit import runtime
    if not runtime.exists():
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__,
                                  "--server.headless", "false"]))
```

Normal `streamlit run` launches skip this block.

### 2. `.vscode/launch.json`: F5 run setup

This adds a run setup called **Run Energy Dashboard (Streamlit)**. It runs `python -m streamlit run app.py` from the `energy-dashboard` folder and opens the browser. A copy is in both `energy-dashboard-group16/.vscode/` and `energy-dashboard/.vscode/`, so it works whichever folder you open in VS Code.

### 3. `.vscode/settings.json`: default Python

This sets the default Python to `C:\Python313\python.exe`, which already had all the packages installed. VS Code only uses this setting if no Python has been picked for the folder yet.

### 4. Packages installed into Python 3.10

VS Code had been using `C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe`, which didn't have the dashboard's packages. That caused this error:

```
ModuleNotFoundError: No module named 'pandas'
```

To fix it, the packages were installed into that Python:

```bash
"C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r requirements.txt
```

## Checked

- Started with Python 3.10 and Python 3.13: the Streamlit server came up at http://localhost:8501 and its health check returned OK.
- On Python 3.10, the dashboard code compiles, and the home page (Overview) loads without errors in a test run without a browser. The other pages weren't tested.

## Troubleshooting

- **`ModuleNotFoundError` on another computer:** run `pip install -r requirements.txt` with the Python VS Code is using. That Python is shown in the bottom-right corner of VS Code.
- **Switching Python:** press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, then pick the one you want.
- **Port 8501 already in use:** another copy of the dashboard is still running. Stop it, or run `streamlit run app.py --server.port 8502`.
