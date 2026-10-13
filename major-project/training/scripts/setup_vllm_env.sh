#!/bin/bash
# One-time setup of a separate vLLM environment (keeps the bitsandbytes env untouched).
# Run on the login node:  bash ~/greenAI/major-project/training/scripts/setup_vllm_env.sh
set -euo pipefail
module load python/3.11.14
uv venv ~/vllm_venv --python 3.11
source ~/vllm_venv/bin/activate
uv pip install vllm nvidia-ml-py "lm-eval[vllm]" git+https://github.com/felipemaiapolo/tinyBenchmarks
# Activation helper used by the session5 job scripts. Reuses HF_TOKEN from the existing env file.
cat > ~/vllm_env.sh <<'EOF'
module load python/3.11.14
HF_LINE=$(grep -E '^export HF_TOKEN=' ~/greenweight_env.sh || true)
[ -n "$HF_LINE" ] && eval "$HF_LINE"
source ~/vllm_venv/bin/activate
EOF
python -c "import vllm, pynvml, lm_eval; print('vllm', vllm.__version__, 'lm_eval', lm_eval.__version__)"
