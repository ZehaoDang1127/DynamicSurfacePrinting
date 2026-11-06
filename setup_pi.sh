#!/usr/bin/env bash
# Run as the ordinary Pi user: bash setup_pi.sh [--camera]
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
WITH_CAMERA=0
WITH_CONVLSTM=0
for arg in "$@"; do
  case "$arg" in
    --camera) WITH_CAMERA=1 ;;
    --convlstm) WITH_CONVLSTM=1; WITH_CAMERA=1 ;;
    -h|--help)
      printf '%s\n' 'Usage: bash setup_pi.sh [--camera] [--convlstm]' \
        'Installs a local Python 3.12 environment for 64-bit Raspberry Pi OS / Debian.' \
        'Requires internet. Uses sudo only for apt packages and optional camera USB rules.' \
        'Default: controller, diagnostics, synthetic capture and NumPy prediction.' \
        '--camera: also install RealSense 2.59.1.11053 and USB permissions.' \
        '--convlstm: camera support plus PyTorch CPU inference/training.'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
if [[ "$(uname -s)" != Linux ]] || [[ "$(uname -m)" != aarch64 && "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'This installer requires Linux ARM64 (aarch64), such as Raspberry Pi OS 64-bit.' >&2
  exit 2
fi
if [[ "$EUID" == 0 ]]; then
  printf '%s\n' 'Run without sudo: bash setup_pi.sh (the script requests sudo only when needed).' >&2
  exit 2
fi
if ! command -v apt-get >/dev/null || ! command -v sudo >/dev/null; then
  printf '%s\n' 'This installer requires Debian/Raspberry Pi OS with apt-get and sudo.' >&2
  exit 2
fi
trap 'printf "Setup stopped at line %s. Fix the reported error, then rerun the same command.\n" "$LINENO" >&2' ERR
packages=(python3 python3-venv ca-certificates)
if (( WITH_CONVLSTM )); then packages+=(libgomp1); fi
if (( WITH_CAMERA )); then packages+=(libusb-1.0-0 curl); fi
printf '%s\n' '[1/4] Installing system prerequisites (sudo may ask for your Pi password).'
sudo apt-get update
sudo apt-get install -y "${packages[@]}"

printf '%s\n' '[2/4] Preparing local Python 3.12 (.venv-pi312).'
if [[ -e .venv-pi312 && ! -x .venv-pi312/bin/python ]]; then
  printf '%s\n' '.venv-pi312 exists but is not a usable Linux environment. Extract into a fresh folder.' >&2
  exit 2
fi
if [[ ! -x .venv-pi312/bin/python ]]; then
  if command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv-pi312
  else
    # Bootstrap uv without modifying the externally managed system Python.
    python3 -m venv .pi-bootstrap
    .pi-bootstrap/bin/python -m pip install --only-binary=:all: 'uv>=0.9,<1'
    # Keep the managed interpreter inside this bundle directory as well.
    export UV_PYTHON_INSTALL_DIR="$ROOT_DIR/.pi-python"
    .pi-bootstrap/bin/python -m uv python install 3.12
    .pi-bootstrap/bin/python -m uv venv --python 3.12 --seed .venv-pi312
  fi
fi
PY="$ROOT_DIR/.venv-pi312/bin/python"
"$PY" -c 'import platform,sys; assert sys.version_info[:2] == (3,12) and platform.machine() in ("aarch64","arm64"), "Expected ARM64 Python 3.12; extract into a fresh folder"; print(sys.executable); print(sys.version)'

printf '%s\n' '[3/4] Installing application dependencies.'
pip_source=()
if [[ -d "$ROOT_DIR/wheelhouse" ]]; then
  pip_source=(--no-index --find-links "$ROOT_DIR/wheelhouse")
fi
"$PY" -m pip install "${pip_source[@]}" --only-binary=:all: -r requirements-pi.txt
if (( WITH_CONVLSTM )); then
  torch_source=()
  if compgen -G "$ROOT_DIR/wheelhouse/torch-*.whl" >/dev/null; then
    torch_source=(--no-index --find-links "$ROOT_DIR/wheelhouse")
  fi
  "$PY" -m pip install "${torch_source[@]}" --only-binary=:all: -r requirements-pi-convlstm.txt
fi
if (( WITH_CAMERA )); then
  "$PY" -m pip install "${pip_source[@]}" --only-binary=:all: -r requirements-pi-camera.txt
  rules_tmp="$(mktemp)"
  trap 'rm -f -- "$rules_tmp"' EXIT
  curl --fail --location --retry 2 --connect-timeout 15 --max-time 120 \
    https://raw.githubusercontent.com/realsenseai/librealsense/v2.59.1/config/99-realsense-libusb.rules \
    --output "$rules_tmp"
  # Check for a rule file before installing it; the downloaded file is not executed.
  if ! grep -q 'SUBSYSTEMS=="usb"' "$rules_tmp"; then
    printf '%s\n' 'Downloaded camera rules are invalid; nothing installed.' >&2
    exit 2
  fi
  target=/etc/udev/rules.d/99-realsense-libusb.rules
  if [[ -e "$target" ]]; then
    sudo cp -p -- "$target" "${target}.backup.$(date +%Y%m%d_%H%M%S_%N)"
  fi
  sudo install -m 0644 -- "$rules_tmp" "$target"
  sudo udevadm control --reload-rules
  printf '%s\n' 'Camera permissions installed. Unplug/reconnect the D405 before using it.'
fi
"$PY" -m pip check
printf '%s\n' '[4/4] Checking installed imports (no printer connection).'
"$PY" -m pi_support.runner doctor
if (( WITH_CAMERA )); then
  "$PY" -c 'import pyrealsense2 as rs; print("RealSense module:", rs.__file__)'
fi
if (( WITH_CONVLSTM )); then
  "$PY" -c 'import torch; print("PyTorch:", torch.__version__); print("CPU check:", torch.ones(2).sum().item())'
fi
printf '\n%s\n' 'Setup complete. Next: bash run_pi.sh self-test' \
  'Read GUIDELINE_中文.md for diagnostic commands and the existing failed-run folder.'
