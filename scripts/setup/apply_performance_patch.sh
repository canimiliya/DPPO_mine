#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMODULE="$PROJECT/source/dppo_v0.6"
PATCH="$PROJECT/reports/PERFORMANCE_PATCH.diff"
EXPECTED='dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc'

test -d "$SUBMODULE/.git" || { echo "Missing submodule: $SUBMODULE" >&2; exit 1; }
test -f "$PATCH" || { echo "Missing patch: $PATCH" >&2; exit 1; }

SHA="$(git -C "$SUBMODULE" rev-parse HEAD)"
test "$SHA" = "$EXPECTED" || {
  echo "Official submodule SHA check failed. Expected $EXPECTED, got $SHA" >&2
  exit 1
}

if git -C "$SUBMODULE" apply --reverse --check -- "$PATCH" >/dev/null 2>&1; then
  echo "Performance patch already applied: $SUBMODULE"
  exit 0
fi

git -C "$SUBMODULE" apply --check -- "$PATCH"
git -C "$SUBMODULE" apply -- "$PATCH"
echo "Performance patch applied: $SUBMODULE"
