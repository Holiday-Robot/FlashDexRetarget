#!/bin/bash
# One-shot, idempotent FlashDexRetarget setup: conda env (python 3.11) with the pinned Isaac
# freeze + learner packages, scripts/env.sh, robot USDs (XHAND, Sharpa Wave), smoke test.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_DIR/scripts/env.sh"
CONDA_ENV=FlashDexRetarget
SKIP_ASSETS=0

usage() {
    cat <<EOF
Usage: bash scripts/setup.sh [--env NAME] [--skip_assets]

  --env NAME        conda env name (default: FlashDexRetarget)
  --skip_assets     do not convert the robot URDFs to USD
EOF
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) CONDA_ENV="$2"; shift 2 ;;
        --env=*) CONDA_ENV="${1#*=}"; shift ;;
        --skip_assets) SKIP_ASSETS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[setup] unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
cd "$PROJECT_DIR"

# ── 1. submodule ────────────────────────────────────────────────────────
echo "[setup] third_party/rsl_rl_flashsac (pinned submodule)"
git submodule update --init third_party/rsl_rl_flashsac

# ── 2. conda env ────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    for p in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$p/etc/profile.d/conda.sh" ] && { source "$p/etc/profile.d/conda.sh"; break; }
    done
fi
command -v conda &>/dev/null || { echo "[setup] conda not found" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "[setup] creating conda env $CONDA_ENV (python 3.11)"
    conda create -n "$CONDA_ENV" python=3.11 -y
    conda activate "$CONDA_ENV"
    pip install --upgrade pip
else
    echo "[setup] conda env $CONDA_ENV exists"
    conda activate "$CONDA_ENV"
fi

# ── 3. packages (all --no-deps: pip cannot resolve the Isaac stack) ─────
pip install "setuptools<81" wheel
pip install --no-build-isolation flatdict==4.0.1   # its setup imports pkg_resources
# The freeze minus the two packages the learner pins newer below (no downgrade/upgrade churn).
FREEZE=$(mktemp)
trap 'rm -f "$FREEZE"' EXIT
grep -v -E '^(rsl-rl-lib|tensordict)==' requirements/isaac_cu128.txt > "$FREEZE"
echo "[setup] Isaac freeze (~15 GB on a fresh env; a no-op when already installed)"
pip install --no-deps --prefer-binary -r "$FREEZE" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.nvidia.com
pip install --no-deps rsl-rl-lib==5.5.1 tensordict==0.13.0 pyvers==0.2.2
pip install --no-deps coacd==1.0.10   # object convex decomposition (src/retarget/object/obj_convex_decompose.py)
pip install --no-deps cuacd==0.1.0    # GPU convex decomposition (convex=cuacd)
pip install --no-deps smplx==0.1.28   # MANO layer for the human demo sources (src/retarget/human_demo)
pip install --no-deps -e third_party/rsl_rl_flashsac
pip install --no-deps -e .

# ── 4. scripts/env.sh (machine-specific, gitignored) ────────────────────
if [ ! -f "$ENV_FILE" ]; then
    echo "[setup] writing scripts/env.sh"
    cat > "$ENV_FILE" <<EOF
# FlashDexRetarget environment (machine-specific, gitignored): conda env, robot USD.
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

export MUJOCO_GL=\${MUJOCO_GL:-disabled}
export OMNI_KIT_ACCEPT_EULA=1
# The xhand_spec robot the flashdexretarget policies are trained on (assets/robot/xhand/urdf/converted_spec).
export FDR_ROBOT_USD=\${FDR_ROBOT_USD:-$PROJECT_DIR/assets/robot/xhand/urdf/converted_spec/xhand_bimanual_spec.usd}
PROJECT_DIR=$PROJECT_DIR
EOF
else
    echo "[setup] scripts/env.sh exists - left as is"
fi

# ── 5. robot USDs (object USDs are made by the retarget pipeline) ──────────
convert_robot() {   # <usd> <convert_assets.py args>
    local usd=$1; shift
    if [ -f "$usd" ]; then
        echo "[setup] $(basename "$usd") exists"
    else
        echo "[setup] converting $(basename "$usd" .usd) URDF to USD"
        MUJOCO_GL=disabled OMNI_KIT_ACCEPT_EULA=1 python src/simulator/isaacsim/convert_assets.py --robot "$@"
    fi
}
if [ "$SKIP_ASSETS" = 1 ]; then
    echo "[setup] --skip_assets: robot USDs not converted"
else
    convert_robot "$PROJECT_DIR/assets/robot/xhand/urdf/converted_spec/xhand_bimanual_spec.usd" \
        --robot-variant bimanual --robot-suffix _spec
    convert_robot "$PROJECT_DIR/assets/robot/sharpa/urdf/converted/sharpa_bimanual.usd" \
        --robot-name sharpa --robot-variant bimanual
fi

# ── 6. smoke test ───────────────────────────────────────────────────────
python - <<'PY'
import importlib.metadata as md

import torch

print(f"[setup] torch {torch.__version__} cuda={torch.cuda.is_available()}")
for p in ("isaacsim", "isaaclab", "rsl-rl-lib", "rsl-rl-flashsac", "tensordict", "FlashDexRetarget"):
    print(f"[setup] {p} {md.version(p)}")
PY
echo "[setup] done. Train with: MOTION=<dataset>/motion.pt bash scripts/train_flashdexretarget.sh (README \"Training\")"
