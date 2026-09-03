#!/usr/bin/env bash
# deploy.sh — DeepSeek V4 Flash Vision Exp two-node TP2 one-command deploy (run on the head node)
# Prereq: two Dell Pro Max with GB10 machines with a 200GbE direct link (RDMA ports configured with static IPs); run this on the head node.
# Usage: ./deploy.sh --head-ip <RDMA_A_IP> --worker-ip <RDMA_B_IP> [--port 8899]
# The worker node must start its worker container first per the Anemll dspark-vllm-gx10 recipe (same image tag).
set -euo pipefail
PORT=8899; HEAD_IP=""; WORKER_IP=""
while [ $# -gt 0 ]; do case "$1" in
  --head-ip) HEAD_IP="$2"; shift 2;;
  --worker-ip) WORKER_IP="$2"; shift 2;;
  --port) PORT="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done
[ -n "$HEAD_IP" ] && [ -n "$WORKER_IP" ] || { echo "usage: $0 --head-ip <RDMA_A_IP> --worker-ip <RDMA_B_IP>"; exit 2; }

IMAGE="ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
REV="86f746b36186f0e567729a5c06a8c918caba82a9"
say() { printf '\033[1m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[deploy] FAIL:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Preflight (pitfalls #1/#5 from this book, mechanized) ──
[ "$(uname -m)" = "aarch64" ] || die "aarch64 required (Dell Pro Max with GB10)"
command -v docker >/dev/null || die "docker required"
if docker ps -a --format '{{.Names}}' | grep -qx dsv4f-head; then
  if docker ps --format '{{.Names}}' | grep -qx dsv4f-head; then
    say "container dsv4f-head already running, skipping to liveness check (idempotent)"; SKIP_START=1
  else
    die "a stopped dsv4f-head container exists. Run docker rm dsv4f-head and retry, or docker start dsv4f-head."
  fi
else SKIP_START=0; fi
[ "${SKIP_START:-0}" = 1 ] || { lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && die "port $PORT is occupied (pitfall #5)"; }
ping -c1 -W2 "$WORKER_IP" >/dev/null || die "worker RDMA IP $WORKER_IP unreachable (ping flags are Linux semantics) — check the QSFP direct link and static IPs"
ENV_FILE="${ENV_FILE:-./serve.env}"   # only scan the env file this recipe specifies, not other .env files in the directory; do not print matched content
if [ -f "$ENV_FILE" ] && grep -E "HOST_IP" "$ENV_FILE" | grep -v '^\s*#' | grep -qE "=(\s*$|.*placeholder|.*head-roce)"; then
  die "$ENV_FILE has unfilled *_HOST_IP placeholders (pitfall #1: ZMQ crash-loop). Fill each with a real value or blank it."
fi

# ── 2. Pull image (tag reference, pitfall #9; skipped on the idempotent path) ──
if [ "${SKIP_START:-0}" = 0 ]; then
  say "pulling image $IMAGE (both machines need it)"
  say "supply-chain note: online deploys can additionally pin the digest; the digest is only lost on offline docker save/load transfers — in that scenario use tag + Image Id assertion across both machines (pitfall #9)"
  docker pull "$IMAGE"
fi

# ── 3. Start head (full production flags; the worker node starts the worker side with the same image per the recipe) ──
say "starting head @ :$PORT (--network host: required for cross-machine NCCL/RDMA; will occupy host ports $PORT and 25000)"
say "weights revision pinned to $REV; first download is large, be patient"
if [ "${SKIP_START:-0}" = 0 ]; then
docker run -d --name dsv4f-head --gpus all --network host \
  -e NCCL_NET=IB -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e DSPARK_MAX_INFLIGHT_PREFILLS=1 -e GB10_HYBRID_NVFP4_M_THRESHOLD=128 \
  -v "$HOME/models:/models" "$IMAGE" \
  --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --revision "$REV" \
  --trust-remote-code --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 \
  --tensor-parallel-size 2 --nnodes 2 --distributed-executor-backend mp \
  --master-addr "$HEAD_IP" --master-port 25000 \
  --max-model-len 1048576 --max-num-seqs 6 --max-num-batched-tokens 8192 \
  --kv-cache-dtype nvfp4_ds_mla --block-size 256 --gpu-memory-utilization 0.835 \
  --enable-prefix-caching --enable-chunked-prefill --long-prefill-token-threshold 1024 --async-scheduling \
  --limit-mm-per-prompt '{"image": 8}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}' \
  --moe-backend flashinfer_b12x --enable-flashinfer-autotune --max-cudagraph-capture-size 42 \
  --port "$PORT" --served-model-name deepseek-v4-flash-vision-exp
fi

# ── 4. Liveness check (two-node load takes a while; real-inference assertion) ──
say "waiting for both nodes to load (can take 20-40 minutes: weight download + cross-machine init)..."
for i in $(seq 1 240); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 15
  [ "$i" = 240 ] && die "not ready after 60 minutes. Check docker logs dsv4f-head; is the worker-side container up?"
done
RESP=$(curl -s -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4-flash-vision-exp","max_tokens":50,
  "chat_template_kwargs":{"thinking":false},
  "messages":[{"role":"user","content":"Reply with exactly one word: DEPLOY_OK"}]}')
echo "$RESP" | grep -q "DEPLOY_OK" || die "inference assertion failed: $(echo "$RESP" | head -c 300)"
say "✅ two-node deploy complete: http://<head>:$PORT/v1"
say "strongly recommended: attach the thinking-tier proxy (pitfall #2): python3 scripts/thinking_proxy.py"
say "remember the three red lines: cold prefill ≤144K (pitfall #4) / concurrency takes short jobs only (pitfall #8) / put flag changes on env default-value lines (pitfall #6)"
