"""
Builds both funds in one go:   python scripts/run_all.py
Add --refresh to force a new Yahoo Finance download (otherwise prices are cached for the day).
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for script in (here / "1" / "serenity.py", here / "2" / "velocity.py"):
    subprocess.run([sys.executable, str(script), *sys.argv[1:]], check=True)
print("\nAll done - open index.html in your browser.")
