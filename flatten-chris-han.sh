#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./flatten-chris-han.sh [-b] [base-ref]
#
# Example:
#   ./flatten-chris-han.sh
#   ./flatten-chris-han.sh -b
#   ./flatten-chris-han.sh origin/main
#
# What it does:
# - auto-detects base as the last commit NOT authored by Chris Han
# - replays commits after <base-ref>
# - keeps non-matching commits as separate commits in original order
# - flattens all matching commits into one final commit on top
# - only creates a new branch when -b is provided
#
# NOTE: this rewrites history. Push with --force-with-lease afterwards.

CREATE_BRANCH=0
if [[ "${1:-}" == "-b" ]]; then
  CREATE_BRANCH=1
  shift
fi

AUTO_STASH_CREATED=0
AUTO_STASH_REF=""

restore_auto_stash() {
  if [[ "$AUTO_STASH_CREATED" -eq 1 ]]; then
    git stash pop --index "$AUTO_STASH_REF" >/dev/null
    AUTO_STASH_CREATED=0
    AUTO_STASH_REF=""
    echo "Restored auto-stashed local changes."
  fi
}

on_exit() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 && "$AUTO_STASH_CREATED" -eq 1 ]]; then
    echo
    echo "Script failed. Your local changes are saved in stash: $AUTO_STASH_REF"
    echo "Restore with:  git stash pop --index $AUTO_STASH_REF"
  fi
}
trap on_exit EXIT

AUTHOR_REGEX='^(Chris Han|chris-han)$'

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
NEW_BRANCH="${CURRENT_BRANCH}-flattened"
SOURCE_HEAD="$(git rev-parse HEAD)"

# Auto-detect base: the first ancestor commit NOT authored by Chris Han.
auto_detect_base() {
  while IFS=' ' read -r hash _rest; do
    local name
    name="$(git show -s --format='%an' "$hash")"
    if ! [[ "$name" =~ $AUTHOR_REGEX ]]; then
      echo "$hash"
      return
    fi
  done < <(git log --format='%H %ae' HEAD)
}

if [[ -n "${1:-}" ]]; then
  BASE_REF="$1"
else
  BASE_REF="$(auto_detect_base)"
fi

if [[ -z "$BASE_REF" ]]; then
  echo "Could not determine base ref (all commits are by Chris Han?)."
  echo "Pass <base-ref> explicitly:  ./flatten-chris-han.sh <base-ref>"
  exit 1
fi

git rev-parse --verify "$BASE_REF" >/dev/null

if ! git diff --quiet || ! git diff --cached --quiet; then
  STASH_MSG="flatten-auto-stash $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git stash push -u -m "$STASH_MSG" >/dev/null
  AUTO_STASH_CREATED=1
  AUTO_STASH_REF="stash@{0}"
  echo "Auto-stashed local changes as $AUTO_STASH_REF"
fi

mapfile -t commits < <(git rev-list --reverse "${BASE_REF}..${SOURCE_HEAD}")

if [[ "${#commits[@]}" -eq 0 ]]; then
  echo "No commits to rewrite between $BASE_REF and $SOURCE_HEAD"
  restore_auto_stash
  exit 0
fi

echo "Base: $BASE_REF"
echo "Commits to process: ${#commits[@]}"
echo

if [[ "$CREATE_BRANCH" -eq 1 ]]; then
  git show-ref --verify --quiet "refs/heads/$NEW_BRANCH" && {
    echo "Branch '$NEW_BRANCH' already exists. Delete it first or use a different name."
    exit 1
  }
  git switch -c "$NEW_BRANCH" "$BASE_REF"
  TARGET_BRANCH="$NEW_BRANCH"
else
  TARGET_BRANCH="$CURRENT_BRANCH"
  git reset --hard "$BASE_REF"
fi

matching_commits=()
matching_first=""
matching_last=""
matching_count=0

for commit in "${commits[@]}"; do
  author="$(git show -s --format='%an' "$commit")"
  subject="$(git show -s --format='%s' "$commit")"

  if [[ "$author" =~ $AUTHOR_REGEX ]]; then
    matching_commits+=("$commit")
    [[ "$matching_count" -eq 0 ]] && matching_first="$commit"
    matching_last="$commit"
    matching_count=$((matching_count + 1))
    echo "Queued Chris Han commit: $commit $subject"
  else
    git cherry-pick "$commit"
    echo "Kept commit: $commit $subject"
  fi
done

if [[ "$matching_count" -gt 0 ]]; then
  for commit in "${matching_commits[@]}"; do
    git cherry-pick --no-commit "$commit"
  done

  git commit -m "Flatten Chris Han commits (${matching_count} commits)

Squashed commits:
${matching_first}..${matching_last}"
fi

restore_auto_stash

echo
echo "Done."
echo "Original branch:  $CURRENT_BRANCH"
echo "Rewritten branch: $TARGET_BRANCH"
echo
echo "If you rewrote a shared branch, push with:"
echo "  git push --force-with-lease origin $TARGET_BRANCH"
