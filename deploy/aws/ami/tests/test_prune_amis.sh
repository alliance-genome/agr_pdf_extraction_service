#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/../prune_amis.sh"
fail=0
check() { if [ "$1" = "$2" ]; then echo "ok: $3"; else echo "FAIL: $3 (got '$1' want '$2')"; fail=1; fi; }

# 4 AMIs oldest->newest; keep 2; protect ami-3 (would-be-pruned but referenced).
LIST=$'2026-07-01 ami-1\n2026-07-02 ami-2\n2026-07-03 ami-3\n2026-07-04 ami-4'
OUT="$(select_amis_to_deregister 2 ami-3 "$LIST" | sort | tr '\n' ' ' | sed 's/ $//')"
# Newest 2 (ami-4, ami-3) kept; ami-3 also protected; so only ami-1, ami-2 pruned.
check "$OUT" "ami-1 ami-2" "keep newest 2, protect referenced"

# Protect a newest AMI (no-op on protection since already kept).
OUT2="$(select_amis_to_deregister 1 ami-4 "$LIST" | sort | tr '\n' ' ' | sed 's/ $//')"
# keep newest 1 (ami-4); prune ami-1,ami-2,ami-3 (none protected among pruned except... ami-4 protected already kept)
check "$OUT2" "ami-1 ami-2 ami-3" "keep newest 1, protected already newest"

# Protect an AMI that WOULD otherwise be pruned (exercises the grep -vx protect branch).
OUT3="$(select_amis_to_deregister 2 ami-1 "$LIST" | sort | tr '\n' ' ' | sed 's/ $//')"
# keep newest 2 (ami-4, ami-3); ami-1 & ami-2 would prune, but ami-1 is protected -> only ami-2.
check "$OUT3" "ami-2" "protect a would-be-pruned AMI"

exit $fail
