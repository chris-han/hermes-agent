#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./flatten-chris-han.sh <base-ref> [author-regex]
#
# Example:
#   ./flatten-chris-han.sh origin/main '^(Chris Han|chris-han)$'
#
# What it does:
# - replays commits after <base-ref> onto a new branch
# - squashes contiguous commits whose author matches [author-regex]
# - preserves non-matching commits as separate commits
#
# NOTE: this rewrites history. Run it on a throwaway branch first.

BASE_REF="${1:-origin/main}"
AUTHOR_REGEX="${2:-^Chris Han$}"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
NEW_BRANCH="${CURRENT_BRANCH}-flattened"

git rev-parse --verify "$BASE_REF" >/dev/null

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree is not clean."
  exit 1
fi

git branch "$NEW_BRANCH" "$BASE_REF"
git checkout "$NEW_BRANCH"

pending=0
pending_first=""
pending_last=""
pending_count=0

flush_pending() {
  if [[ "$pending" -eq 1 ]]; then
    git commit -m "Flatten Chris Han commits (${pending_count} commits)

Squashed commits:
${pending_first}..${pending_last}"
    pending=0
    pending_first=""
    pending_last=""
    pending_count=0
  fi
}

while IFS= read -r commit; do
  author="$(git show -s --format='%an' "$commit")"
  subject="$(git show -s --format='%s' "$commit")"

  if [[ "$author" =~ $AUTHOR_REGEX ]]; then
    git cherry-pick --no-commit "$commit"
    if [[ "$pending" -eq 0 ]]; then
      pending=1
      pending_first="$commit"
    fi
    pending_last="$commit"
    pending_count=$((pending_count + 1))
    echo "Queued Chris Han commit: $commit $subject"
  else
    flush_pending
    git cherry-pick "$commit"
    echo "Kept commit: $commit $subject"
  fi
done < <(git rev-list --reverse "${BASE_REF}..${CURRENT_BRANCH}")

flush_pending

echo
echo "Done."
echo "Original branch:  $CURRENT_BRANCH"
echo "Rewritten branch: $NEW_BRANCH"