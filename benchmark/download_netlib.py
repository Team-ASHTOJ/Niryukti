#!/usr/bin/env python3
"""Fetch a small Netlib smoke suite from a pinned public HiGHS test-data mirror.

These are benchmark data, not solver implementation code. Re-fetching is optional;
the small demonstration suite is cached locally for offline use.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
REVISION='73cac48c5340d775a477087198611862559be250'
BASE=f'https://raw.githubusercontent.com/ERGO-Code/HiGHS/{REVISION}/check/instances/'
p=argparse.ArgumentParser();p.add_argument('--output',default='datasets');p.add_argument('instances',nargs='*',default=['afiro','adlittle','israel','e226']);a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=True);manifest=[]
for name in a.instances:
    if not name.replace('-','').replace('_','').isalnum():raise ValueError('Invalid instance name')
    url=BASE+name+'.mps'
    try:data=urllib.request.urlopen(url,timeout=30).read()
    except Exception as exc:
        manifest.append(dict(instance=name,source=url,status='DOWNLOAD_ERROR',message=str(exc)));print(name,exc);continue
    path=root/(name+'.mps')
    digest=hashlib.sha256(data).hexdigest();path.write_bytes(data)
    manifest.append(dict(instance=name,path=path.name,source=url,sha256=digest,bytes=len(data),suite='Netlib LP',mirror_revision=REVISION))
    print(name,len(data),digest)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
