# 🚀 Enabling Real Training Mode (Gsplat)

## 🛑 Blockers Identified
1.  **CUDA**: Found `v13.0` (Configured ✅).
2.  **Build Tools**: `ninja` & `cmake` (Installed by Agent ✅).
3.  **Compiler**: `cl.exe` (Microsoft C++ Compiler) **MISSING ❌**.

## 📋 The Solution: "What you need to provide"

You need to provide the **C++ Environment**.
The agent cannot install Visual Studio for you.

### Step 1: Install Visual Studio Build Tools
1.  Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2.  Run the installer.
3.  **CRITICAL**: Select the workload **"Desktop development with C++"**.
4.  Ensure the "MSVC ... C++ x64/x86 build tools" component is checked on the right.
5.  Install (approx 2-3 GB).

### Step 2: Run the Install Command
Once VS is installed, open the **"Developer Command Prompt for VS 2022"** (search for it in Windows Start Menu).
Then run:

```powershell
cd "e:\SEM 3\EL\Glimpse3D"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;" + $env:Path
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

### Verification
Run `python -c "import gsplat; print('SUCCESS')"` to confirm.
Then run `python ai_modules/gsplat/train.py ...` for real training.
