#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

usage() {
    cat <<'EOF'
usage: scripts/train.sh <command> [args]

commands:
  bc [args...]          Run behavior cloning
  bc-resume [args...]   Resume behavior cloning from latest BC checkpoint
  ppo [ckpt] [args...]  Run PPO, defaulting to latest BC checkpoint
  tensorboard           Start TensorBoard for logs/
EOF
}

latest_bc_checkpoint() {
    local latest
    latest="$(find "$ROOT_DIR/checkpoints" -maxdepth 1 -name 'bc_epoch*.pt' -printf '%f\n' 2>/dev/null | sort | tail -n 1)"
    if [[ -z "$latest" ]]; then
        echo "No BC checkpoints found in $ROOT_DIR/checkpoints" >&2
        return 1
    fi
    printf '%s\n' "$ROOT_DIR/checkpoints/$latest"
}

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$ROOT_DIR/scripts/setup_env.sh"
fi

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
    usage
    exit 2
fi
shift

case "$cmd" in
    bc)
        python -m scripts.train_bc "$@"
        ;;
    bc-resume)
        python -m scripts.train_bc --resume "$(latest_bc_checkpoint)" "$@"
        ;;
    ppo)
        ckpt="${1:-}"
        if [[ -n "$ckpt" && "$ckpt" != --* ]]; then
            shift
        else
            ckpt="$(latest_bc_checkpoint)"
        fi
        python -m scripts.train_rl --bc-ckpt "$ckpt" "$@"
        ;;
    tensorboard|tb)
        tensorboard --logdir logs
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
