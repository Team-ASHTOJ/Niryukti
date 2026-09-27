#!/usr/bin/env python3
"""Cache public benchmark data with immutable-source/checksum records, not solver code."""
import argparse, concurrent.futures, gzip, hashlib, json, urllib.request
from pathlib import Path
NETLIB_REV='f1cc423067407d55d579c9c35fb01edf860dbc24'
def fetch(url):
    with urllib.request.urlopen(url,timeout=30) as r:
        data=r.read(50_000_001)
        if len(data)>50_000_000: raise ValueError('Download exceeds 50 MB policy')
        return data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['netlib','miplib','qplib'],required=True)
    p.add_argument('--output',type=Path,default=Path('datasets/public'))
    p.add_argument('instances',nargs='*');a=p.parse_args();root=a.output/a.suite;root.mkdir(parents=True,exist_ok=True)
    if a.suite=='netlib':
        index=json.loads(fetch(f'https://api.github.com/repos/coin-or-tools/Data-Netlib/git/trees/{NETLIB_REV}'))['tree']
        available=[v['path'][:-7] for v in index if v['path'].endswith('.mps.gz')]
        names=a.instances or sorted(available)
        base=f'https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/{NETLIB_REV}/'
        sources=[(n,base+n+'.mps.gz') for n in names]
        (root/'UPSTREAM-LICENSE').write_bytes(fetch(base+'LICENSE'))
    elif a.suite=='qplib':
        names=a.instances or ['9002','8938','8792','8515']
        sources=[(n,'https://qplib.zib.de/qplib/QPLIB_'+n+'.qplib') for n in names]
    else:
        # Explicit manageable starter campaign, not the entire 240-instance suite.
        names=a.instances or ['markshare1','markshare2','flugpl','neos-911880','enlight_hard']
        sources=[(n,'https://miplib.zib.de/WebData/instances/'+n+'.mps.gz') for n in names]
    for n,_ in sources:
        if not n.replace('-','').replace('_','').replace('.','').isalnum():raise ValueError('Invalid name')
    def download(item):
        n,url=item
        record=dict(instance=n,suite=a.suite,source=url,source_revision=NETLIB_REV if a.suite=='netlib' else None)
        try:
            data=fetch(url)
            if a.suite!='qplib':data=gzip.decompress(data)
            if len(data)>250_000_000:raise ValueError('Expanded model exceeds 250 MB policy')
            path=root/(n+('.qplib' if a.suite=='qplib' else '.mps'));path.write_bytes(data)
            record.update(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),status='CACHED')
        except Exception as e:record.update(status='DOWNLOAD_ERROR',message=str(e))
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(download,sources))
    (root/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(suite=a.suite,selected=len(records),cached=sum(v['status']=='CACHED' for v in records),manifest=str(root/'manifest.json'))))
if __name__=='__main__':main()
