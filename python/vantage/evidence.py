"""Offline solve bundles: integrity hashes plus fresh numerical verification.

Hashes are not signatures. Retain the returned archive digest outside the bundle
to detect replacement of both payloads and their manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import zipfile
from . import _binary, solve

def digest(data): return hashlib.sha256(data).hexdigest()
def encoded(data): return json.dumps(data,indent=2,allow_nan=False).encode()

def sign_bundle(path, private_key, signature_path, *, openssl='openssl'):
    """Detached SHA-256 signature using a caller-managed RSA/EC PEM key.

    Private keys are never embedded. Encrypted keys must be unlocked externally.
    """
    data=Path(path).read_bytes()
    with tempfile.TemporaryDirectory(prefix='vantage-sign-') as temp:
        snapshot=Path(temp)/'bundle.zip';snapshot.write_bytes(data)
        proc=subprocess.run([openssl,'dgst','-sha256','-sign',str(Path(private_key).resolve()),str(snapshot)],
                            stdin=subprocess.DEVNULL,capture_output=True,timeout=30)
        if proc.returncode:raise ValueError('Signing failed: '+proc.stderr.decode(errors='replace'))
        with Path(signature_path).open('xb') as f:f.write(proc.stdout)
    return dict(signature=str(signature_path),sha256=digest(data),scope='Detached signature; key trust is managed by the recipient.')

def verify_signed_bundle(path, signature_path, trusted_public_key, *, binary=None, openssl='openssl'):
    """Require a separately trusted public key, then perform numerical checking."""
    with tempfile.TemporaryDirectory(prefix='vantage-signed-check-') as temp:
        snapshot=Path(temp)/'bundle.zip';snapshot.write_bytes(Path(path).read_bytes())
        signature=Path(temp)/'signature.bin';signature.write_bytes(Path(signature_path).read_bytes())
        proc=subprocess.run([openssl,'dgst','-sha256','-verify',str(Path(trusted_public_key).resolve()),
                             '-signature',str(signature),str(snapshot)],capture_output=True,timeout=30)
        if proc.returncode:raise ValueError('Signature verification failed for the supplied trusted key')
        result=verify_bundle(snapshot,binary=binary)
        result['signature_verified']=True
        result['scope']='Signature accepted under caller-supplied key; mathematical verification remains separate. Integer tree proof is not replayed.'
        return result

def _verify(binary, model, solution):
    p=subprocess.run([binary,'verify',str(model),str(solution)],capture_output=True,text=True,timeout=120)
    try: result=json.loads(p.stdout)
    except ValueError: result=dict(status='VERIFICATION_PROCESS_ERROR',message=p.stderr)
    return dict(exit_code=p.returncode,result=result,stderr=p.stderr)

def create_bundle(model, output, *, binary=None, **options):
    binary=str(Path(shutil.which(_binary(binary)) or _binary(binary)).resolve())
    binary_hash=digest(Path(binary).read_bytes())
    with tempfile.TemporaryDirectory(prefix='vantage-bundle-') as temp:
        folder=Path(temp)
        source=Path(model)
        snapshot=folder/('input'+source.suffix);snapshot.write_bytes(source.read_bytes())
        native=folder/'model.json'
        conversion=subprocess.run([binary,'convert',str(snapshot),str(native)],capture_output=True,text=True)
        if conversion.returncode: raise ValueError(conversion.stderr)
        result=solve(native,binary=binary,**options)
        solution=folder/'solution.json';solution.write_bytes(encoded(result))
        verification=_verify(binary,native,solution)
        if digest(Path(binary).read_bytes())!=binary_hash: raise RuntimeError('Binary changed during measurement')
        files={'model.json':native.read_bytes(),'source.model':snapshot.read_bytes(),'solution.json':solution.read_bytes(),
               'verification.json':encoded(verification),
               'metadata.json':encoded(dict(binary_sha256=binary_hash,options=options,platform=platform.platform(),
                   solver_version=result.get('version'),source_format=source.suffix,source_sha256=digest(snapshot.read_bytes()),
                   scope='Numerical original-space verification; integer tree proof is not replayed. SHA-256 integrity is not a signature.'))}
        manifest=encoded(dict(format='vantage-evidence-1',sha256={name:digest(data) for name,data in files.items()}))
        # Exclusive creation prevents silently replacing previous evidence.
        with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,data in files.items():archive.writestr(name,data)
            archive.writestr('manifest.json',manifest)
    return dict(path=str(output),sha256=digest(Path(output).read_bytes()),
                solve_status=result.get('status'),verification=verification['result'].get('status'))

def verify_bundle(path, *, binary=None, expected_sha256=None):
    path=Path(path)
    actual=digest(path.read_bytes())
    if expected_sha256 and actual!=expected_sha256: raise ValueError('Archive SHA-256 mismatch')
    expected={'model.json','source.model','solution.json','verification.json','metadata.json','manifest.json'}
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist())!=expected or len(archive.namelist())!=len(expected):
            raise ValueError('Unexpected or duplicate archive members')
        if sum(info.file_size for info in archive.infolist())>256*1024*1024:
            raise ValueError('Bundle exceeds 256 MiB verification limit')
        files={name:archive.read(name) for name in expected}
    manifest=json.loads(files.pop('manifest.json'))
    if manifest.get('format')!='vantage-evidence-1' or set(manifest.get('sha256',{}))!=set(files):
        raise ValueError('Invalid evidence manifest')
    for name,data in files.items():
        if digest(data)!=manifest['sha256'][name]:raise ValueError('Payload SHA-256 mismatch: '+name)
    # Never run an executable or command supplied by the archive.
    with tempfile.TemporaryDirectory(prefix='vantage-verify-bundle-') as temp:
        folder=Path(temp)
        for name in ('model.json','solution.json'):(folder/name).write_bytes(files[name])
        verification=_verify(_binary(binary),folder/'model.json',folder/'solution.json')
    accepted=verification['exit_code']==0 and verification['result'].get('status') in ('VERIFIED_OPTIMAL','VERIFIED_FEASIBLE','VERIFIED_INFEASIBLE')
    return dict(integrity_checked=True,archive_digest_pinned=bool(expected_sha256),sha256=actual,
                numerically_verified=accepted,verification=verification,
                scope='Integer feasibility is not a replayed optimality proof; hashes do not authenticate the author.')

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    create=sub.add_parser('create');create.add_argument('model');create.add_argument('output');create.add_argument('--binary');create.add_argument('--method',default='auto');create.add_argument('--device',default='cpu')
    verify=sub.add_parser('verify');verify.add_argument('bundle');verify.add_argument('--binary');verify.add_argument('--expected-sha256')
    verify.add_argument('--signature');verify.add_argument('--public-key')
    sign=sub.add_parser('sign');sign.add_argument('bundle');sign.add_argument('signature');sign.add_argument('--private-key',required=True)
    args=parser.parse_args()
    if args.action=='create': result=create_bundle(args.model,args.output,binary=args.binary,method=args.method,device=args.device)
    elif args.action=='sign':result=sign_bundle(args.bundle,args.private_key,args.signature)
    else:
        if bool(args.signature)!=bool(args.public_key):parser.error('--signature and --public-key must be supplied together')
        if args.signature:
            if args.expected_sha256 and digest(Path(args.bundle).read_bytes())!=args.expected_sha256:raise ValueError('Archive SHA-256 mismatch')
            result=verify_signed_bundle(args.bundle,args.signature,args.public_key,binary=args.binary)
        else:result=verify_bundle(args.bundle,binary=args.binary,expected_sha256=args.expected_sha256)
    print(json.dumps(result,indent=2))
    if args.action=='verify' and not result['numerically_verified']:raise SystemExit(2)

if __name__=='__main__':main()
