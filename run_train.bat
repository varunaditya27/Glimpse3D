call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set TORCH_NVCC_FLAGS=-allow-unsupported-compiler
python ai_modules/gsplat/train.py assets/input_images/input.png --out ai_modules/gsplat/output/user_run --iter 100
