"""ctypes interface to the independently implemented C++ core; no subprocess."""
import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
from .planning import SolverSession

class NativeSession(SolverSession):
    """Same updates/solve interface as SolverSession, through the C ABI.

    Native requests are serialized by the library. Close handles explicitly.
    """
    def __init__(self, model, *, library=None):
        suffix='dylib' if sys.platform=='darwin' else 'so'
        default=Path(__file__).resolve().parents[2]/'build'/f'libvantage_c.{suffix}'
        from ._runtime import shared_library
        self._lib=ctypes.CDLL(shared_library(library))
        self._lib.vantage_abi_version.restype=ctypes.c_uint
        if self._lib.vantage_abi_version()!=1: raise RuntimeError('Unsupported native ABI')
        self._lib.vantage_open.argtypes=[ctypes.c_char_p];self._lib.vantage_open.restype=ctypes.c_void_p
        self._lib.vantage_close.argtypes=[ctypes.c_void_p];self._lib.vantage_close.restype=None
        self._lib.vantage_request.argtypes=[ctypes.c_void_p,ctypes.c_char_p];self._lib.vantage_request.restype=ctypes.c_char_p
        self._lib.vantage_last_error.restype=ctypes.c_char_p
        with tempfile.TemporaryDirectory(prefix='vantage-native-') as folder:
            path=model
            if isinstance(model,dict) or hasattr(model,'data'):
                path=Path(folder)/'model.json';path.write_text(json.dumps(model.data if hasattr(model,'data') else model,allow_nan=False))
            self._handle=self._lib.vantage_open(os.fsencode(path))
        if not self._handle: raise ValueError(self._lib.vantage_last_error().decode())
    def _request(self, action, **fields):
        if not self._handle: raise RuntimeError('Native session is closed')
        raw=self._lib.vantage_request(self._handle,json.dumps(dict(action=action,**fields),allow_nan=False).encode())
        if raw is None: raise ValueError(self._lib.vantage_last_error().decode())
        return json.loads(raw)
    def close(self):
        if self._handle:self._lib.vantage_close(self._handle);self._handle=None
