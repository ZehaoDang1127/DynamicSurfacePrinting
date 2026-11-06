#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PI_PYTHON:-}" ]]; then
    PYTHON_BIN="$PI_PYTHON"
elif [[ -x "$ROOT_DIR/.venv-pi312/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv-pi312/bin/python"
elif [[ -x "$ROOT_DIR/.venv-controller/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv-controller/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
    echo "Python environment not found. Run: bash setup_pi.sh" >&2
    echo "Or select an existing environment: PI_PYTHON=/absolute/path/bin/python bash run_pi.sh doctor" >&2
    exit 2
fi

if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
    echo "PI_PYTHON must point to a working Python 3.10+ interpreter." >&2
    exit 2
fi
exec "$PYTHON_BIN" -m pi_support.runner "$@"
