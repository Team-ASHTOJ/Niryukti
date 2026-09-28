"""Installed command forwards arguments to the bundled native executable."""
import subprocess
import sys
from ._runtime import executable

def main():
    return subprocess.call([executable(), *sys.argv[1:]])
