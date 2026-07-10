#!/usr/bin/env bash
set -euo pipefail

[[ -f "${BASH_SOURCE[0]}" ]] || {
  echo "install.sh must run from a local file" >&2
  exit 2
}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${ROOT}/pyproject.toml" && -d "${ROOT}/src/amd_ai" ]] || {
  echo "incomplete toolkit checkout: ${ROOT}" >&2
  exit 2
}
command -v python3.12 >/dev/null || {
  echo "python3.12 is required" >&2
  exit 2
}
NON_INTERACTIVE=0
for argument in "$@"; do
  [[ "${argument}" == "--non-interactive" ]] && NON_INTERACTIVE=1
done
if [[ "${NON_INTERACTIVE}" -eq 0 ]]; then
  [[ -t 0 && -t 1 ]] || {
    echo "interactive install requires a terminal" >&2
    exit 2
  }
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3.12 -m amd_ai.installer.bootstrap --source-root "${ROOT}" "$@"
