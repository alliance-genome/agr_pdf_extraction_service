#!/usr/bin/env bash
# Keep runtime version constraints authoritative without installing ML packages.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
runtime_constraints="$(mktemp)"
trap 'rm -f "$runtime_constraints"' EXIT
# pip constraints reject extras (e.g. celery[redis]); retain the package/version.
sed -E 's/^([[:alnum:]_.-]+)\[[^]]+\]/\1/' requirements.txt > "$runtime_constraints"
python -m pip install -c "$runtime_constraints" -r requirements-test.txt
