import sys
import traceback

print("Attempting to import backend.app.main...")
try:
    from backend.app import main
    print("Import SUCCESS")
except Exception:
    print("Import FAILED")
    traceback.print_exc()
