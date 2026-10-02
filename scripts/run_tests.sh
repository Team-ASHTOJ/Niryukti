#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ctest --test-dir "${1:-build}" --output-on-failure
python3 tests/test_cli.py --binary "${1:-build}/vantage"
python3 tests/test_profiles.py
VANTAGE_BINARY="${1:-build}/vantage" python3 tests/test_planning.py
case "$(uname -s)" in
  Darwin) native_suffix=dylib ;;
  *) native_suffix=so ;;
esac
# ctypes loads the instrumented library into an ordinary Python interpreter.
# Preload its runtime only for native Python tests; standalone C++ sanitizer
# tests above retain their normal leak checks. CPython allocations are outside
# the native-library leak oracle, so disable leak reporting for these hosts.
asan_runtime=""
cxx_runtime=""
if command -v ldd >/dev/null 2>&1; then
    asan_runtime=$(ldd "${1:-build}/libniryukti_c.$native_suffix" 2>/dev/null |         awk '/libasan.*=>/ {print $3; exit}' || true)
    cxx_runtime=$(ldd "${1:-build}/libniryukti_c.$native_suffix" 2>/dev/null | \
        awk '/libstdc[+][+].*=>/ {print $3; exit}' || true)
fi
native_python() {
    if [[ -n "$asan_runtime" ]]; then
        LD_PRELOAD="$asan_runtime${cxx_runtime:+:$cxx_runtime}${LD_PRELOAD:+:$LD_PRELOAD}"             ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_leaks=0" python3 "$@"
    else
        python3 "$@"
    fi
}
VANTAGE_BINARY="${1:-build}/vantage" VANTAGE_LIBRARY="${1:-build}/libvantage_c.$native_suffix" native_python tests/test_extensions.py
VANTAGE_BINARY="${1:-build}/vantage" python3 tests/test_trust_and_replanning.py
VANTAGE_TEST_BINARY="${1:-build}/vantage" python3 tests/test_benchmark.py
VANTAGE_TEST_BINARY="${1:-build}/vantage" python3 dashboard/tests/test_dashboard.py
NIRYUKTI_BINARY="${1:-build}/niryukti" python3 tests/test_service.py
python3 scripts/check_dependencies.py
VANTAGE_BINARY="${1:-build}/vantage" NIRYUKTI_LIBRARY="${1:-build}/libniryukti_c.$native_suffix" native_python tests/test_certificates.py

NIRYUKTI_BINARY="${1:-build}/niryukti" python3 tests/test_conflict_learning.py

NIRYUKTI_BINARY="${1:-build}/niryukti" python3 tests/test_engine_checkpoints.py

NIRYUKTI_BINARY="${1:-build}/niryukti" python3 tests/test_analysis.py

NIRYUKTI_BINARY="${1:-build}/niryukti" python3 tests/test_advanced.py
