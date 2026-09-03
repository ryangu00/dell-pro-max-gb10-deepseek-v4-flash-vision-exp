# DeepSeek V4 Flash Vision Exp on Dell Pro Max with GB10 ×2 (vLLM TP2)

> 双机张量并行部署 DeepSeek V4 Flash Vision Exp(社区视觉增强权重),1M 上下文、原生图像输入、6 座并发。
> 目标:照着做可复现,并绕开我们踩过的 9 个坑。所有数字为我们环境的实测值,附测试口径。

## 硬件与版本(全 pin)

| 项 | 规格/版本 |
|---|---|
| 机器 | Dell Pro Max with GB10 ×2(GB10 芯片,128GB 统一内存 each,sm_121) |
| 互联 | 200GbE QSFP 双机直连(RoCE/RDMA,无需交换机) |
| 驱动 | NVIDIA 580.142;`TORCH_CUDA_ARCH_LIST=12.1a` |
| 部署配方 | Anemll `dspark-vllm-gx10` 镜像 `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`(内含 vLLM 0.25.2.dev fork g752a3a5;registry 路径即拉取入口) |
| 权重 | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` @ revision `86f746b36186f0e567729a5c06a8c918caba82a9`(原生 ViT) |

双机直连要点:两台机 QSFP 口对插一根 DAC/AOC 线,RDMA 口设静态 IP(如 `<RDMA_A_IP>`/`<RDMA_B_IP>`),`NCCL_NET=IB`。**双机 TP2 不需要 InfiniBand 交换机**;管理面走普通 2.5GbE 即可。

## 结果速览(实测,口径见括号与 BENCHMARKS 节)

| 指标 | 数值 |
|---|---|
| 并发吞吐 | 75.8 tok/s 聚合(6 并发短 prompt 混测,temperature 0.3,输出 200-800 tok) |
| 上下文 | 1,048,576(1M) |
| KV 容量 | NVFP4 KV cache 总池 ≈233 万 token(**全部座位共享**,非每座 1M×6;验证:vLLM 启动日志 `GPU KV cache size` 行) |
| 长上下文 | 95K needle 单针检索命中(needle 置于文首/文中/文尾各测) |
| 视觉 | 原生 image_url;图/prompt 上限由服务端参数 `--limit-mm-per-prompt '{"image": 8}'` 决定(超出报 400,上限可自行调整) |
| 冷 prefill | 120-144K 多次实测正常;**≥200K 两次触发主机级重启**(详见坑 #4) |

## 快速开始

**一键部署(head 机)**:`scripts/deploy.sh --head-ip <RDMA_A_IP> --worker-ip <RDMA_B_IP>`(预检直连+占位符检查→拉镜像→完整参数启动→真实推理断言);worker 机按配方起同镜像 worker 侧。手动流程:

两机各起一个容器(head/worker),镜像 pin `0.1.1`。以下是我们生产实例的**完整 vLLM serving 参数**(取自启动日志,可整体照抄进配方的 serve env/启动命令):

```
--model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --revision 86f746b36186f0e567729a5c06a8c918caba82a9
--trust-remote-code --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4
--tensor-parallel-size 2 --nnodes 2 --distributed-executor-backend mp --master-addr <RDMA_A_IP> --master-port 25000
--max-model-len 1048576 --max-num-seqs 6 --max-num-batched-tokens 8192
--kv-cache-dtype nvfp4_ds_mla --block-size 256 --gpu-memory-utilization 0.835
--enable-prefix-caching --enable-chunked-prefill --long-prefill-token-threshold 1024 --async-scheduling
--limit-mm-per-prompt '{"image": 8}'
--speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}'
--moe-backend flashinfer_b12x --enable-flashinfer-autotune --max-cudagraph-capture-size 42
--port 8899 --served-model-name deepseek-v4-flash-vision-exp
```

容器 env 关键项:`DSPARK_MAX_INFLIGHT_PREFILLS=1`(防并发大 prefill 互踩)、`GB10_HYBRID_NVFP4_M_THRESHOLD=128`、`NCCL_NET=IB`、`TORCH_CUDA_ARCH_LIST=12.1a`。

```bash
# 验活(head 机)
curl -s http://<NODE_A_IP>:8899/v1/models
```

> 注:`nvfp4_ds_mla`/DSpark 投机解码是该配方 fork 的能力,上游原版 vLLM 未必同名——请 pin 镜像 0.1.1 复现,换版本先小流量验证。

### thinking 分档代理(强烈建议)

此权重默认开思考,裸用会踩坑 #2。方案:一个 133 行纯 stdlib 的端口分档 proxy(源码在 [`scripts/thinking_proxy.py`](scripts/thinking_proxy.py)):

- `:8901` → 注入 `chat_template_kwargs: {"thinking": false}` = 快档(日常/agent)
- `:8902` → 保持思考 = 质量档(复杂推理)

调用方按端口选档,零客户端改造。实测代价:开思考 6-13× 响应时间。收益:在我们的 agentic 评测集上,关思考档出现的语义错误(条件理解反转、相对日期算错)在开思考档**未再复现**——建议按任务类型分档,不要全局一刀切。

## 避坑清单(9 个,均为一手事故)

1. **配方 .env 占位符 crash-loop**:配方 `.env` 模板深处有 `*_HOST_IP` 类占位符(如 `head-roce-ip`),不清零就启动 → ZMQ `No such device` 循环崩(我们撞了 56 次自动重启才定位)。**启动前 `grep -rn "HOST_IP" .env*`,占位符全部填真值或清空。**
2. **默认思考吃光 max_tokens**:此栈服务端默认 `default_chat_template_kwargs: {thinking: true, reasoning_effort: max}`(启动日志可证),思考 token 会吃满 max_tokens → `content=None` 且不报错。客户端必传 `chat_template_kwargs: {"thinking": false}`(或用上面的分档 proxy)。同类预算口径见 BENCHMARKS 节。
3. **双机 TP2 内存模型与单机不同**:TP2 每机只加载一半权重——单机加载超大权重的 wedge(统一内存耗尽假死)及其对策(大 swapfile+watermark 调参)**不适用也不需要**,别照抄单机教程。
4. **冷 prefill 超大 prompt 打死主机**:无 prefix cache 时一次性喂 ≥200K prompt,我们两次观察到**主机级重启**(非进程级;驱动 580.142+此镜像组合)。120-144K 多次实测正常。长文靠 prefix cache 复用,冷启动别贪。
5. **端口被占的静默假死**:症状像"服务没起来",实为端口被其他服务占用。起服务前对 8899(vLLM)与 8901/8902(分档 proxy)三个端口逐一 `lsof -iTCP:<port> -sTCP:LISTEN` 检查——我们被一个第三方服务占过预定端口。
6. **serving 参数直写 env 文件的 `${VAR:=}` 默认值行**:配方 install 脚本重跑会重写自动生成段——写在生成段的自定义参数会全部丢失。改配方 `.env`(容器工作目录下 install 脚本引用的那份)里 `${VAR:=default}` 默认值行本身才能存活。
7. **ssh 远程 pkill -f 自杀**:pattern 出现在 ssh 命令行文本里时 pkill 连 ssh 会话一起杀。用 `pgrep` 拿 pid 再 kill,或 pattern 用 `[x]` 正则拆分。
8. **并发只发短活**:冷的大 prompt 混进并发,聚合吞吐从 75.8 掉到 ~8 tok/s(6 并发同measure)。长上下文任务走单座队列,并发座位只发短请求。
9. **镜像用 tag 引用,别 digest-pin 搬运**:`docker save | load` 到离线机会丢 repo digest,digest 引用直接失效。tag 引用+两机 `Image Id` 一致性断言。

## BENCHMARKS(口径)

- **并发**:6 路同时,短 prompt(1-4K)混合,temperature 0.3,输出 200-800 tok,取稳态聚合吞吐。
- **needle**:95K 上下文单针插入(文首/中/尾各一轮),问答判定命中。
- **思考双档**:reasoning 模型做质量对比时,agentic 类题目建议开/关思考各测一档——我们实测同一道题关思考档答错、开思考档答对(条件理解与日期推算类),只测关思考会系统性低估。
- **双预算**:调用带思考的模型,`reasoning` 需单独 cap+总 max_tokens 给足。反例实测:16K 总预算被思考吃掉 16000、正文 0 字(请求仅设 `max_tokens:16000`,未 cap reasoning)。
- 长上下文标称值不可直接信任,**在自己的栈上实测安全线**——同硬件另一模型标称 1M、实测 95K 确定性崩(该案例详见本系列 Flash-Next 篇,随系列后续发布)。

## 何时选这套方案

✅ 需要 1M 上下文+视觉+并发的自托管;✅ 有两台Dell Pro Max with GB10
❌ 只有单机 → 本系列 Qwen3.8-27B 单机篇;❌ 预算敏感短上下文 → 本系列 GPT-OSS-120B ollama 篇

---
*RyanAI Lab · 数字来自我们的常驻环境实测,更新于 2026-09。欢迎 issue 反馈。*
