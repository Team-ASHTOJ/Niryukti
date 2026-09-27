#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ctest --test-dir "${1:-build}" --output-on-failure
python3 tests/test_cli.py --binary "${1:-build}/vantage"
python3 tests/test_profiles.py
VANTAGE_TEST_BINARY="${1:-build}/vantage" python3 tests/test_benchmark.py
VANTAGE_TEST_BINARY="${1:-build}/vantage" python3 dashboard/tests/test_dashboard.py
python3 scripts/check_dependencies.py
