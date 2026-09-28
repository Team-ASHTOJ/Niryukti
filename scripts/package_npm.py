#!/usr/bin/env python3
"""Stage a bounded source distribution; never include datasets or run artifacts."""
import argparse
from pathlib import Path
import shutil
import subprocess
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
stage=out/'npm-source'
if stage.exists(): raise SystemExit('Staging directory already exists; choose a new output directory')
shutil.copytree(root/'packaging/npm',stage)
for name in ('LICENSE','NOTICE'):
    shutil.copy2(root/name,stage/name)
engine=stage/'engine';engine.mkdir()
for name in ('src','include','app','third_party'):
    shutil.copytree(root/name,engine/name)
shutil.copy2(root/'CMakeLists.txt',engine/'CMakeLists.txt')
subprocess.run(['npm','pack',str(stage),'--pack-destination',str(out)],check=True)
