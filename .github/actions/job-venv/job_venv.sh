#!/usr/bin/env bash
# The job's venv, reused across runs (action.yml).
#
# Kept under $NBA_OU_VENV_DIR (default ~/.cache/nba_ou/venvs), one directory
# per runner, then one venv per requirements file and Python:
#
#   <root>/<runner>/<requirements name>-<key>/
#
# where <key> hashes the requirements file, the Python interpreter and its
# version, and LAYOUT below. A venv is reused only when it finished building
# (.complete) and its installed packages are still those it was built with
# (.freeze): anything else is deleted and built again. A venv no run has used
# for PRUNE_DAYS days is deleted. The repo itself is reinstalled (editable,
# --no-deps) only when pyproject.toml or the checkout path changes.
#
# The venv is built in place, not built elsewhere and moved: a venv's scripts
# carry its absolute path in their shebangs.
set -euo pipefail
shopt -s nullglob

: "${REQUIREMENTS:?REQUIREMENTS must name the requirements file of the job}"
# Bump to rebuild every venv, e.g. when this script changes what a venv holds.
LAYOUT=1
PRUNE_DAYS=14

venv_phase_started=$SECONDS
log_timing() {
    printf '[time] job venv %s: %ss\n' "$1" "$((SECONDS - venv_phase_started))"
    venv_phase_started=$SECONDS
}

echo "[start] job venv: waiting for cache lock"
root="${NBA_OU_VENV_DIR:-$HOME/.cache/nba_ou/venvs}/${RUNNER_NAME:-local}"
mkdir -p "$root"
exec 9>"$root/.lock"
flock 9
log_timing "cache lock"

echo "[start] job venv: checking cache for $REQUIREMENTS"
python_id="$(python -c 'import sys; print(sys.executable, sys.version)')"
key="$(
    {
        echo "layout=$LAYOUT"
        echo "$python_id"
        cat "$REQUIREMENTS"
    } | sha256sum | cut -c1-16
)"
name="$(basename "$REQUIREMENTS" .txt)-$key"
venv="$root/$name"

for old in "$root"/*/; do
    old="${old%/}"
    [[ "$old" == "$venv" ]] && continue
    if [[ -z "$(find "$old" -maxdepth 1 -name .last-used -mtime "-$PRUNE_DAYS")" ]]; then
        echo "Deleting $(basename "$old"): unused for $PRUNE_DAYS days"
        rm -rf "$old"
    fi
done

freeze() {
    "$venv/bin/python" -m pip freeze --all --exclude-editable --disable-pip-version-check
}

if [[ -f "$venv/.complete" ]]; then
    if [[ "$(freeze)" == "$(cat "$venv/.freeze")" ]]; then
        echo "Reusing $name"
    else
        echo "$name holds other packages than it was built with; rebuilding"
        rm -rf "$venv"
    fi
elif [[ -e "$venv" ]]; then
    echo "$name did not finish building; rebuilding"
    rm -rf "$venv"
fi
log_timing "cache validation and pruning"

if [[ ! -e "$venv" ]]; then
    echo "[start] job venv: installing $REQUIREMENTS"
    echo "Building $name"
    python -m venv "$venv"
    "$venv/bin/python" -m pip install --upgrade pip
    "$venv/bin/python" -m pip install -r "$REQUIREMENTS"
    freeze >"$venv/.freeze"
    touch "$venv/.complete"
    log_timing "dependency installation"
fi
touch "$venv/.last-used"

echo "[start] job venv: checking the editable repo install"
editable="$(
    {
        echo "$GITHUB_WORKSPACE"
        cat pyproject.toml
    } | sha256sum | cut -c1-16
)"
if [[ "$(cat "$venv/.editable" 2>/dev/null)" != "$editable" ]]; then
    "$venv/bin/python" -m pip install --no-deps -e .
    echo "$editable" >"$venv/.editable"
else
    echo "The repo is already installed in $name"
fi
log_timing "editable repo installation"

echo "$venv/bin" >>"$GITHUB_PATH"
