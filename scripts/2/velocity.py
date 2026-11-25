"""Velocity Fund - run from anywhere: python scripts/2/velocity.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # par84 of this folder
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))  # fundlib, config
import par84
import fundlib
from config import VELOCITY

if __name__ == "__main__":
    fundlib.run(VELOCITY, par84, refresh="--refresh" in sys.argv)
