# Windows setup — step by step

For Windows 10/11 with VS Code, Git and Python. Takes about 15–20 minutes, mostly downloads. Run every command in **PowerShell** (VS Code: *Terminal → New Terminal*).

## 0. Check what you have

```powershell
python --version     # need 3.10–3.13; 3.12 recommended
git --version
code --version       # VS Code
```

- **No Python, or the wrong version:** install Python 3.12 from https://www.python.org/downloads/windows/ and tick **"Add python.exe to PATH"** in the installer.
- **`python` opens the Microsoft Store:** turn off *Settings → Apps → Advanced app settings → App execution aliases → python.exe*, then reopen PowerShell.
- **Several versions installed:** use `py -3.12` instead of `python` in step 2.

## 1. Get the project

Unzip `supplychain-lens-phase1.zip` into your projects folder (for example `C:\Users\<you>\Projects\`), then:

```powershell
cd C:\Users\<you>\Projects\supplychain-lens
code .
```

Optional, for the audit script and for studying the original:

```powershell
git clone https://github.com/ahmedzainsyed/nexus-scct.git reference/nexus-scct
```

`reference/` is git-ignored, so it never gets committed.

## 2. Create the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

- **Error "running scripts is disabled on this system":** run this once, then activate again:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

VS Code: press `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.\.venv\Scripts\python.exe`. New terminals then activate it automatically. When VS Code offers to install the recommended extensions (Python, Ruff, Jupyter, Mermaid preview), accept.

## 3. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## 4. Install PyTorch (CPU build) first

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This downloads about 200–300 MB. Using the CPU index on purpose avoids a multi-GB GPU build: SupplyChain Lens never needs a GPU (ADR-0009).

**Have an NVIDIA GPU and want to use it?** It's optional. Pick the matching CUDA command at https://pytorch.org/get-started/locally/ instead. Everything also works on CPU.

## 5. Install the rest

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

`requirements-dev.txt` includes the training and app packages plus pytest and ruff. `-e .` installs the project in *editable* mode, so `import supplychain_lens` works from anywhere and picks up your edits.

## 6. Verify

```powershell
python scripts/smoke_test.py
pytest
ruff check .
streamlit run app/Home.py
```

Expected:

| Command | You should see |
|---|---|
| `python scripts/smoke_test.py` | Five numbered sections, `✓ learning`, `max \|ONNX − PyTorch\|` around 1e-7 `✓`, a sizing estimate, and `✓ PASSED` in about 20–40 seconds |
| `pytest` | `15 passed` |
| `ruff check .` | `All checks passed!` |
| `streamlit run app/Home.py` | A browser tab at http://localhost:8501 with the SupplyChain Lens home page. Stop it with `Ctrl+C` |

The smoke test writes `reports/hardware_profile.json`. The home page shows it under *Training-machine profile*, and Phases 5–6 read it to size the models.

## 7. First commit and GitHub

```powershell
git init -b main
git add .
git status            # check: no .venv\, no reference\, no reports\hardware_profile.json
git commit -m "chore: Phase 1 - SupplyChain Lens scaffold, environment, CI and smoke test"
```

Create an empty repository on GitHub called `supplychain-lens` (*New repository*; no README, no license, no .gitignore — the project already has them). Then:

```powershell
git remote add origin https://github.com/<your-username>/supplychain-lens.git
git push -u origin main
```

Open the repository's **Actions** tab. The *CI* workflow should finish green in a few minutes. It runs two jobs: the full suite with CPU PyTorch, and an app-only job that proves the dashboard works without PyTorch.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install torch` says "No matching distribution" | Your Python is too new or 32-bit. Install 64-bit Python 3.12 |
| `ModuleNotFoundError: supplychain_lens` | Activate `.venv` and run `python -m pip install -e .` from the repo root |
| `streamlit` not recognised | The venv isn't active; run `.\.venv\Scripts\Activate.ps1` |
| Smoke test is very slow (> 2 min) | Close other heavy apps and plug in the laptop (power-saving mode throttles the CPU); send me the numbers it prints |
| Antivirus slows the first `import torch` | Normal on the first run only |
