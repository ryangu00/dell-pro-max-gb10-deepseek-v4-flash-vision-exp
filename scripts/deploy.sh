#!/usr/bin/env bash
# deploy.sh — DeepSeek V4 Flash Vision Exp 双机 TP2 一键部署(head 机运行)
# 前提:两台 Dell Pro Max with GB10 已 200GbE 直连(RDMA 口静态 IP 配好),本脚本在 head 机跑。
# 用法: ./deploy.sh --head-ip <RDMA_A_IP> --worker-ip <RDMA_B_IP> [--port 8899]
# worker 机需先按 Anemll dspark-vllm-gx10 配方起 worker 容器(同镜像 tag)。
set -euo pipefail
PORT=8899; HEAD_IP=""; WORKER_IP=""
while [ $# -gt 0 ]; do case "$1" in
  --head-ip) HEAD_IP="$2"; shift 2;;
  --worker-ip) WORKER_IP="$2"; shift 2;;
  --port) PORT="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done
[ -n "$HEAD_IP" ] && [ -n "$WORKER_IP" ] || { echo "用法: $0 --head-ip <RDMA_A_IP> --worker-ip <RDMA_B_IP>"; exit 2; }

IMAGE="ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
REV="86f746b36186f0e567729a5c06a8c918caba82a9"
say() { printf '\033[1m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[deploy] FAIL:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. 预检(本书坑 #1/#5 的机器化) ──
[ "$(uname -m)" = "aarch64" ] || die "需 aarch64(Dell Pro Max with GB10)"
command -v docker >/dev/null || die "需要 docker"
if docker ps -a --format '{{.Names}}' | grep -qx dsv4f-head; then
  if docker ps --format '{{.Names}}' | grep -qx dsv4f-head; then
    say "容器 dsv4f-head 已在跑,跳到验活(幂等)"; SKIP_START=1
  else
    die "存在已停止的 dsv4f-head 容器。docker rm dsv4f-head 后重跑,或 docker start dsv4f-head。"
  fi
else SKIP_START=0; fi
[ "${SKIP_START:-0}" = 1 ] || { lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && die "端口 $PORT 被占(坑 #5)"; }
ping -c1 -W2 "$WORKER_IP" >/dev/null || die "worker RDMA IP $WORKER_IP 不可达(ping 参数为 Linux 语义)——检查 QSFP 直连与静态 IP"
ENV_FILE="${ENV_FILE:-./serve.env}"   # 只扫本配方指定 env,不碰目录里其他 .env;不打印匹配内容
if [ -f "$ENV_FILE" ] && grep -E "HOST_IP" "$ENV_FILE" | grep -v '^\s*#' | grep -qE "=(\s*$|.*placeholder|.*head-roce)"; then
  die "$ENV_FILE 存在未填的 *_HOST_IP 占位符(坑 #1:ZMQ crash-loop)。逐个填真值或清空。"
fi

# ── 2. 拉镜像(tag 引用,坑 #9;幂等路径跳过) ──
if [ "${SKIP_START:-0}" = 0 ]; then
  say "拉镜像 $IMAGE(两机都要)"
  say "供应链提示:在线部署可再 pin digest 加固;仅当离线 docker save/load 搬运时 digest 会丢失,那种场景用 tag+两机 Image Id 断言(坑 #9)"
  docker pull "$IMAGE"
fi

# ── 3. 启动 head(完整生产参数;worker 机按配方以同镜像起 worker 侧) ──
say "启动 head @ :$PORT(--network host:跨机 NCCL/RDMA 需要;将占用宿主 $PORT 与 25000 端口)"
say "权重 revision pin $REV;首次下载量大,耐心"
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

# ── 4. 验活(双机加载耗时长;真实推理断言) ──
say "等双机加载(可达 20-40 分钟:权重下载+跨机初始化)..."
for i in $(seq 1 240); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 15
  [ "$i" = 240 ] && die "60 分钟未就绪。docker logs dsv4f-head 排查;worker 侧容器是否已起?"
done
RESP=$(curl -s -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4-flash-vision-exp","max_tokens":50,
  "chat_template_kwargs":{"thinking":false},
  "messages":[{"role":"user","content":"回复一个词:DEPLOY_OK"}]}')
echo "$RESP" | grep -q "DEPLOY_OK" || die "推理断言失败: $(echo "$RESP" | head -c 300)"
say "✅ 双机部署完成: http://<head>:$PORT/v1"
say "强烈建议接 thinking 分档 proxy(坑 #2):python3 scripts/thinking_proxy.py"
say "记住三条红线:冷 prefill ≤144K(坑 #4)/并发只发短活(坑 #8)/参数改 env 默认值行(坑 #6)"
