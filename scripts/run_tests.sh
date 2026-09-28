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
VANTAGE_BINARY="${1:-build}/vantage" VANTAGE_LIBRARY="${1:-build}/libvantage_c.$native_suffix" python3 tests/test_extensions.py
VANTAGE_BINARY="${1:-build}/vantage" python3 tests/test_trust_and_replanning.py
VANTAGE_TEST_BINARY="${1:-build}/vantage" python3 tests/test_benchmark.py
VANTAGE_TEST_BINARY="${1:-build}/vantage" python3 dashboard/tests/test_dashboard.py
NIRYUKTI_BINARY="${1:-build}/niryukti" python3 tests/test_service.py
python3 scripts/check_dependencies.py
