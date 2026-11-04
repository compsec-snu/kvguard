#!/usr/bin/env bash
# Build the per-engine Python environments.
#
# Usage: scripts/setup-envs.sh [--lock] [--recreate] [NAME...]
#
#   NAME        A directory under scripts/envs/: tools (pytest, ruff, numpy), vllm-0.32.0,
#               sglang-0.5.22. Default: all of them.
#   --lock      Re-resolve requirements.in into requirements.lock before installing. This can
#               change pins; commit the new lock together with the reason.
#   --recreate  Delete ENVS_DIR/NAME before installing.
#
# Environment variables:
#   ENVS_DIR        Where environments are created (default: envs; ignored by Git).
#   PYTHON_VERSION  Interpreter version (default: 3.12).
#   UV              The uv executable (default: uv).
#
# Needs network access (pypi.org, pypi.nvidia.com, download.pytorch.org) and roughly 10 GB of
# disk per environment; check `df -h` first. Idempotent: an existing environment is synchronized
# to its lock, so re-running after a lock change updates it in place. Each environment gets
# kvg-manifest.json and kvg-freeze.txt describing exactly what was installed.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

ENVS_DIR=${ENVS_DIR:-envs}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
UV=${UV:-uv}

do_lock=0
recreate=0
names=()
while (($#)); do
    case $1 in
        --lock) do_lock=1 ;;
        --recreate) recreate=1 ;;
        -h|--help)
            awk 'NR > 1 { if ($0 ~ /^#/) { sub(/^# ?/, ""); print } else exit }' "$0"
            exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) names+=("$1") ;;
    esac
    shift
done
if ((${#names[@]} == 0)); then
    for dir in scripts/envs/*/; do
        names+=("$(basename -- "$dir")")
    done
fi

if ! command -v "$UV" >/dev/null 2>&1; then
    echo "uv not found (install: https://docs.astral.sh/uv/)" >&2
    exit 1
fi

for name in "${names[@]}"; do
    spec=scripts/envs/$name
    lock=$spec/requirements.lock
    env=$ENVS_DIR/$name
    if [[ ! -f $spec/requirements.in ]]; then
        echo "[$name] no specification at $spec/requirements.in" >&2
        exit 1
    fi
    uv_args=()
    if [[ -f $spec/uv-args ]]; then
        mapfile -t uv_args < <(grep -vE '^[[:space:]]*(#|$)' "$spec/uv-args" || true)
    fi

    if ((do_lock)) || [[ ! -f $lock ]]; then
        echo "[$name] resolving $spec/requirements.in -> $lock"
        "$UV" pip compile --quiet --python-version "$PYTHON_VERSION" "${uv_args[@]}" \
            -o "$lock" "$spec/requirements.in"
    fi

    if ((recreate)) && [[ -d $env ]]; then
        echo "[$name] removing $env"
        rm -rf -- "$env"
    fi
    if [[ ! -d $env ]]; then
        echo "[$name] creating $env (Python $PYTHON_VERSION)"
        "$UV" venv --quiet --python "$PYTHON_VERSION" "$env"
    fi

    echo "[$name] synchronizing $env to $lock"
    "$UV" pip sync --quiet --python "$env/bin/python" "${uv_args[@]}" "$lock"

    # Decision 0006 item 7: the kvguard package itself is installed editable after the sync, so a
    # re-sync (which makes the environment match the lock exactly) does not remove it.
    if [[ -f pyproject.toml ]]; then
        echo "[$name] installing kvguard editable (--no-deps)"
        "$UV" pip install --quiet --python "$env/bin/python" --no-deps -e .
    fi

    "$UV" pip freeze --python "$env/bin/python" > "$env/kvg-freeze.txt"
    python3 - "$name" "$env" "$lock" "$UV" "${uv_args[@]}" > "$env/kvg-manifest.json" <<'PY'
import datetime, hashlib, json, platform, subprocess, sys

name, env, lock, uv = sys.argv[1:5]
uv_args = sys.argv[5:]

def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()

print(json.dumps({
    "name": name,
    "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "host": platform.node(),
    "platform": platform.platform(),
    "python": run(f"{env}/bin/python", "--version"),
    "uv": run(uv, "--version"),
    "lock": lock,
    "lock_sha256": hashlib.sha256(open(lock, "rb").read()).hexdigest(),
    "uv_args": uv_args,
}, indent=2))
PY
    echo "[$name] ready: $env"
done
