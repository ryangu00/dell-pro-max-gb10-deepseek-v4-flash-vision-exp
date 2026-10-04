# DeepSeek V4 Flash Vision Exp on Dell Pro Max with GB10 ×2 (vLLM TP2)

> Two-node tensor-parallel deployment of DeepSeek V4 Flash Vision Exp (community vision-enhanced weights): 1M context, native image input, 6-seat concurrency.
> Goal: follow along and reproduce, while dodging the 9 pitfalls we hit. Most numbers are measured; estimates and observations are labeled.

## Hardware and versions (fully pinned)

| Item | Spec/version |
|---|---|
| Machine | Dell Pro Max with GB10 ×2 (GB10 chip, 128GB unified memory each, sm_121) |
| Interconnect | 200GbE QSFP direct link between the two nodes (RoCE/RDMA, no switch needed) |
| Driver / platform | NVIDIA 580.142 on the pre-2026-09-18 platform (DGX OS OTA 7.4.0, kernel 6.17.0-1014); `TORCH_CUDA_ARCH_LIST=12.1a`. Everything measured for the original `0.1.1` image was measured on that platform. For the 2026-09 and 2026-10 updates, refer to each section for the applicable stack, platform, and whether a number is measured, observed, or estimated. |
| Deploy recipe | Anemll `dspark-vllm-gx10` image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (bundles vLLM 0.25.2.dev fork g752a3a5; the registry path is the pull entry point) |
| Weights | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` @ revision `86f746b36186f0e567729a5c06a8c918caba82a9` (native ViT) |

Direct-link essentials: plug one DAC/AOC cable between the two machines' QSFP ports, give the RDMA ports static IPs (e.g. `<RDMA_A_IP>`/`<RDMA_B_IP>`), set `NCCL_NET=IB`. **Two-node TP2 does NOT need an InfiniBand switch**; the management plane runs fine over plain 2.5GbE.

## Results at a glance (measured; setup in parentheses and in the BENCHMARKS section)

| Metric | Value |
|---|---|
| Concurrent throughput | 75.8 tok/s aggregate (6 concurrent short-prompt mix, temperature 0.3, 200-800 tok output) |
| Context | 1,048,576 (1M) |
| KV capacity | NVFP4 KV cache total pool ≈2.33M tokens (**shared across all seats**, not 1M×6 per seat; verify against the `GPU KV cache size` line in the vLLM startup log) |
| Long context | 95K needle single-needle retrieval hit (needle placed at start/middle/end of the document, one run each) |
| Vision | Native image_url; images-per-prompt cap set by the server flag `--limit-mm-per-prompt '{"image": 8}'` (exceeding it returns 400; raise the cap as needed) |
| Cold prefill | 120-144K fine across multiple measured runs on the stack documented here (image 0.1.1, driver 580.142). On that stack a cold prefill of 200K tokens or more was treated as a host-reboot hazard (see pitfall #4, qualified there). On a different stack and platform on 2026-09-25, cold prefills of about 200K, 400K, 700K and 950K tokens all completed (one run each); the hazard was not reproduced there (see Update 2026-10). |

## Update 2026-10 — what changed since the 2026-09-20 update

Labels used in this section: **M** = measured, **O** = observed once, **E** = estimated. Counts are shown where relevant; e.g. "O, n=2" means two observed events.

### How the two Dell Pro Max with GB10 nodes have been used since the 2026-09-20 update (all dates local)

- **2026-09-18 (O):** both nodes updated, one at a time (about 20 minutes each, no failures): DGX OS OTA 7.4.0 to 7.6.0, kernel 6.17.0-1014 to 7.0.0-1019-nvidia, driver 580.142 to 580.178.04, kernel option `kho=off` added (it avoids a memory-registration error that hit multi-node RDMA on kernel 7.0 with the default setting), new firmware.
- **2026-09-24 (O):** the image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` was deleted from both nodes in a disk cleanup (18.8 GB each; it had been superseded by the community recipe image on 2026-09-17).
- **about 2026-09-25 01:10–01:14 (O):** the Qwen3.8-Flash-Next 1M setup on the two nodes was demoted to a rollback tier and the nodes went back to serving the DeepSeek V4 Flash Vision-Exp weights, two-node tensor-parallel 2, on the community recipe image (vLLM 0.1.dev20759, nightly build of 2026-09-13), as the long-context path and fallback; the fast interactive first hop moved off the GB10 nodes to another machine.
- **about 2026-09-26 07:30 (O):** the nodes were reorganized again: one node now serves a single-node build of Vision-Exp (EXL3 K2.2-D2 quantization, 256,000-token context, vision kept), and the other node was freed for other work. Credits for that build: recipe repository tpurtell/ds4-mia-exl3-k2-1spark and weights wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1.
- Nothing in this README's own recipe has been run in production since 2026-09-17.

### Cold-prefill ladder, 2026-09-25 (answers the line-27 and pitfall-4 hazard)

- **Conditions (M):** Vision-Exp weights on the community recipe image (vLLM 0.1.dev20759, nightly of 2026-09-13), two Dell Pro Max with GB10 nodes, tensor-parallel 2 over RDMA, kernel 7.0.0-1019-nvidia, driver 580.178.04, DGX OS 7.6.0. Requests sent directly to the engine, streamed, thinking off, temperature 0, max_tokens 32, random-word filler so the prefix cache could not help, a short secret planted at a depth of 0.185 to 0.528, health check after each tier, abort threshold 1.5 GiB of available memory. One run per tier.
- **Results (M):**

| Prompt tokens | First token | Prefill | Recall | Lowest available memory (node 1 / node 2) |
|---|---|---|---|---|
| 199,838 | 102.9 s | 1,942 tok/s | exact | 5.50 / 3.92 GiB |
| 399,720 | 245.2 s | 1,630 tok/s | exact | 5.34 / 3.87 GiB |
| 699,290 | 519.6 s | 1,346 tok/s | exact | 5.13 / 3.86 GiB |
| 949,221 | 815.2 s | 1,164 tok/s | exact | 4.94 / 3.83 GiB |

Engine healthy after every tier; verdict pass. Available memory barely moved during the run because the KV cache is reserved at startup.

- **What this does not show:** the stack, kernel, driver, firmware, KV precision and draft settings all differ from the original stack at once, so the change that made the difference (if any) is not isolated; the original image was not re-tested on the new platform; one run per tier, thinking off, no concurrency; the original reboot events were never diagnosed. The hazard was not reproduced on a different stack and platform. It is not claimed to be fixed.

### Passing the ladder does not mean "open the gateway to 1M"

- **Gateway timeout reality (O):** with a 300 s default request timeout at the OpenAI-compatible gateway in front of the nodes, first-token times of 519.6 s (700K) and 815.2 s (950K) exceed it, and at 400K (245.2 s) there is little margin; a single non-streaming request of 8,192 output tokens at 32.8 tok/s needs about 250 s of generation on top of prefill (E, calculated). Client libraries also enforce idle timeouts of 300 s on streams (measured with a probe on 2026-09-25: a stream that stays silent for 310 s fails at 301.5 s; one silent for 290 s succeeds). A long prefill sends no bytes. So on 2026-09-25 we raised the gateway's per-model input cap from 131,072 to 262,144 tokens, not to 1M. At 262,144 the first token is about 150 s (E, interpolated between the 200K and 400K tiers). The cap is a capacity limit, not a latency guarantee.
- **Token-count mismatch (O):** token counts differ between the gateway and the engine: for one Chinese-heavy document the engine's tokenizer counted 239,804 tokens and the gateway's pre-check counted 286,378 (ratio 1.194, about 19% more). One sample; do not use 1.194 as a general conversion factor across languages, code or tool schemas.

### Unresolved failure mode: thinking-max long generation hangs the two-node engine

- **Observation (O, n=2):** on the community recipe image above (not on the image pinned in this README, which was not tested under this load), tensor-parallel 2, the same weights: at maximum reasoning effort with max_tokens 32,768 and 2 concurrent requests, the engine hung twice, on 2026-09-19 and 2026-09-25. Observed signature on 2026-09-25: generation throughput fell to 0 (1 request running, 5,912 tokens computed, 559 generated), a shared-memory broadcast warning appeared every 60 s, about 5 minutes later an RPC call to sampling timed out and the engine died; the API port refused connections while the container stayed "Up", the first worker held about 106 GB and the GPU showed about 96% utilization. `/health` kept returning 200 during the hang until the engine died. Kernel logs on both nodes showed no out-of-memory or GPU Xid events. A non-thinking run on the same engine over about 2.3 hours had zero errors.
- **Investigation status (O, 2026-09-25):** checked and ruled out on the nodes: missing sm_121 support in the build, NCCL version (only 2.31.2 loaded), adaptive mode (off by default), the upstream shared-memory notification fix (already in the image), GPU errors during serving, and memlock limits. Kernel 7.0.0-1019 is at most a weak suspect (its known fault is an error at startup, which did not occur). A community survey on 2026-09-25 found the same signature reported by others with no root cause and no fix. The investigation (baseline reproduction and candidate A/B runs) was parked unexecuted. **Root cause unknown.** Mitigation we applied: maximum-effort requests were removed from the fallback chains that share this engine.
- **Single-node non-reproduction (O):** on 2026-09-26 the same kind of load (maximum effort, max_tokens 32,768, 2/4/6 concurrent streams) ran for 102 minutes on the single-node EXL3 build without a stall or preemption. This is one session on a different engine and quantization and does not identify the cause.

### Pointers

- **Unresolved:** (1) The two original reboot events (2026-08/09) have no retained logs in our records; trigger conditions, the memory state at the time, and the kernel at the time are unconfirmed. (2) Whether the hazard is gone is unknown; it was not reproduced once per tier on another stack and platform, and it has not been tested on this README's image with kernel 7.0.0-1019 and driver 580.178.04. (3) The thinking-max hang: root cause unknown, n=2, not reproduced deliberately, no tested fix; treat maximum-effort long generation on a two-node tensor-parallel setup of this model as a known failure mode. (4) The README's measured numbers (75.8 tok/s aggregate, 95K needle, KV 2.33M tokens) have not been repeated on the current platform. (5) The 262,144 first-token time (about 150 s) is interpolated, not measured; the 8,192-token output time is calculated. (6) The gateway-versus-engine token ratio (1.194) is a single sample. (7) Whether the pinned image can still be pulled from its public registry: not checked for this update.
- A separate write-up is planned for the single-node EXL3 K2.2-D2 vision build.

## Quick start

**One-command deploy (head node)**: `scripts/deploy.sh --head-ip <RDMA_A_IP> --worker-ip <RDMA_B_IP>` (preflight for direct link + placeholder check → pull image → launch with full flags → real-inference assertion); on the worker node, start the worker side with the same image per the recipe. Manual flow:

Run one container per machine (head/worker), image pinned to `0.1.1`. Below are the **complete vLLM serving flags** of the production instance we ran on this stack until 2026-09-17 (taken from the startup log; safe to copy wholesale into the recipe's serve env/launch command):

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

Key container env vars: `DSPARK_MAX_INFLIGHT_PREFILLS=1` (keeps concurrent large prefills from trampling each other), `GB10_HYBRID_NVFP4_M_THRESHOLD=128`, `NCCL_NET=IB`, `TORCH_CUDA_ARCH_LIST=12.1a`.

```bash
# Liveness check (head node)
curl -s http://<NODE_A_IP>:8899/v1/models
```

> Note: `nvfp4_ds_mla`/DSpark speculative decoding are capabilities of this recipe's fork — upstream vanilla vLLM may not have them under these names. Pin image 0.1.1 to reproduce; if you switch versions, validate with a small traffic sample first.

### Thinking-tier proxy (strongly recommended)

These weights ship with thinking on by default; using them bare walks straight into pitfall #2. Our fix: a 133-line pure-stdlib port-per-tier proxy (source in [`scripts/thinking_proxy.py`](scripts/thinking_proxy.py)):

- `:8901` → injects `chat_template_kwargs: {"thinking": false}` = fast tier (daily/agent work)
- `:8902` → keeps thinking = quality tier (complex reasoning)

Callers pick a tier by port — zero client changes. Measured cost: thinking on = 6-13× response time. Measured payoff: on our agentic eval set, the semantic errors that show up with thinking off (inverted condition understanding, wrong relative-date math) **never reproduced** with thinking on. Split by task type; don't force one global setting.

## Pitfalls (9, all firsthand incidents)

1. **Recipe .env placeholder crash-loop**: deep inside the recipe's `.env` template are `*_HOST_IP`-style placeholders (e.g. `head-roce-ip`). Launch without clearing them → ZMQ `No such device` crash loop (it took us 56 auto-restarts to pin down). **Before launch: `grep -rn "HOST_IP" .env*` and fill every placeholder with a real value or blank it.**
2. **Default thinking eats all of max_tokens**: this stack's server defaults to `default_chat_template_kwargs: {thinking: true, reasoning_effort: max}` (verifiable in the startup log). Thinking tokens fill max_tokens → `content=None` with no error. Clients must send `chat_template_kwargs: {"thinking": false}` (or use the tier proxy above). Same budget methodology in the BENCHMARKS section.
3. **Two-node TP2 memory model differs from single-node**: with TP2 each machine loads only half the weights — the single-node wedge from loading oversized weights (unified memory exhaustion hang) and its countermeasures (big swapfile + watermark tuning) **do not apply and are not needed**. Don't copy single-node tutorials blindly.
4. **Huge cold prefill was a host-reboot hazard on this stack (2026-08/09)**: feeding a prompt of 200K tokens or more in one shot with no prefix cache was recorded as rebooting the whole host (not just the process) on driver 580.142 and this image. Kernel 6.17.0-1014 was the platform kernel for this stack, but the kernel, memory state and trigger conditions at the time of each reboot event are unconfirmed. The same symptom (a single cold 256K prefill) is reported in an open issue on the launcher's upstream repository as of 2026-08-20. We did not keep logs of our own reboot events. 120-144K was fine across multiple measured runs. **Qualification (2026-10):** on a different stack (a community recipe image with vLLM 0.1.dev20759, tensor-parallel 2, same weights) and a different platform (kernel 7.0.0-1019, driver 580.178.04), cold prefills of 200K to 950K tokens completed once each (Update 2026-10). That is a non-reproduction on another stack, not a fix; we did not re-test this README's image on the new platform. Long documents should still lean on prefix-cache reuse.
5. **Silent dead-looking service from an occupied port**: symptoms look like "the service never came up", but the port was grabbed by another service. Before launch, check all three ports — 8899 (vLLM) and 8901/8902 (tier proxy) — one by one with `lsof -iTCP:<port> -sTCP:LISTEN`. A third-party service once squatted on our reserved port.
6. **Where to put custom serving flags so they survive**: the recipe's install script rewrites its auto-generated config section on every re-run — any custom flags placed in that generated section are wiped. The only edits that survive are changes to the `${VAR:=default}` default-value lines in the recipe's own `.env` (the copy the install script reads, in the container working directory). Edit the defaults, not the generated output.
7. **Remote `pkill -f` suicide over ssh**: when the pattern appears in the ssh command-line text, pkill kills the ssh session along with the target. Use `pgrep` to get the pid then kill, or break the pattern with a `[x]` regex.
8. **Concurrency takes short jobs only**: mixing a cold large prompt into the concurrent load dropped aggregate throughput from 75.8 to ~8 tok/s (same 6-concurrency measurement). Route long-context work through a single-seat queue; concurrent seats get short requests only.
9. **Reference images by tag; don't digest-pin for transfer**: `docker save | load` onto an offline machine drops the repo digest, so digest references break outright. Use tag references plus an `Image Id` equality assertion across both machines.

## BENCHMARKS (methodology)

- **Concurrency**: 6 simultaneous streams, short-prompt (1-4K) mix, temperature 0.3, 200-800 tok output, steady-state aggregate throughput.
- **Needle**: 95K-context single-needle insertion (one round each at start/middle/end), hit judged by Q&A.
- **Thinking two-tier**: when quality-benchmarking a reasoning model, run agentic-type questions with thinking on AND off — we measured the same question answered wrong with thinking off and right with thinking on (condition understanding and date arithmetic). Testing only the off tier systematically underestimates the model.
- **Dual budget**: when calling a thinking model, cap `reasoning` separately AND give total max_tokens enough headroom. Measured counterexample: a 16K total budget had 16,000 tokens eaten by thinking and 0 characters of content (the request set only `max_tokens:16000` with no reasoning cap).
- Advertised long-context figures cannot be trusted at face value — **measure the safe line on your own stack**. On the same hardware, another model advertised 1M and crashed at 95K on its August recipe (a later recipe ran 200K and 1M tiers without a crash; see the 2026-09-20 update below).

## When to pick this setup

✅ Self-hosting that needs 1M context + vision + concurrency; ✅ you have two Dell Pro Max with GB10
❌ Single machine only → this series' Qwen3.8-27B single-node installment; ❌ budget-sensitive short context → this series' GPT-OSS-120B ollama installment

---
*RyanAI Lab · Numbers are measured unless labeled (O = observed, E = estimated). Updated 2026-10. Issues welcome.*

## Update 2026-09-17 — we no longer run this exact stack in production

Numbers measured for the original `0.1.1` image reproduced as written when we last ran it (before the 2026-09-18 platform update; the image was deleted from our nodes on 2026-09-24 and has not been re-run since) (image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, weights pinned). But after two months two things surfaced that we could not fix inside this image:

1. **Any image request could hang the engine.** A single 1024×768 PNG produced an NCCL collective timeout on rank 1; rank 0 stayed up, `/health` kept returning 200, and every request timed out until a full restart. Text-only workloads never see this.
2. **The prefix-cache fix merged upstream on 2026-09-03 was never released** into a new image, so long sessions always paid a cold prefill.

We ran a three-way A/B against two community stacks with a frozen decision rule and moved production. The harness, the numbers (decode / cold prefill / 6-stream / `tool-eval-bench` hardmode / vision OCR / KV pool, four columns including the pre-TP2 two-node layout) and the twelve pitfalls are in a separate repo:

**→ [dell-pro-max-gb10-vllm-stack-ab](https://github.com/ryangu00/dell-pro-max-gb10-vllm-stack-ab)**

What this repo is still good for: the fastest-booting, largest-KV-pool (≈2.33 M tokens) stack of the three for **text-only** DeepSeek-V4-Flash serving on a GB10 pair, and the only one of the three that runs without any patch. If you do not send images and do not depend on prefix-cache hits, it remains a valid choice. This was measured on the pre-2026-09-18 platform and has not been re-tested since.

## Update 2026-09-20: this stack is no longer our production

On 2026-09-20 we moved production off DeepSeek V4 Flash Vision-Exp (two-node vLLM TP2) onto **Qwen3.8-Flash-Next** on the same two Dell Pro Max with GB10 nodes. New engine, as run: weights `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (QAD mixed-precision NVFP4), the community two-node cluster recipe (vLLM b12x MoE backend, TP2 over RoCE), `--max-model-len 1000000` via static YaRN factor 4, fp8 KV cache, MTP-4 with `--no-async-scheduling`, `--tool-call-parser qwen3_coder --reasoning-parser qwen3`, thinking mode on (reasoning effort `low` on the fast tier, `medium` on the quality tier). The new engine also serves the old model name as an alias, so **no client configuration changed** (endpoint and model name are the same; the only behavioural difference is that responses now carry the thinking text in a `reasoning` field of the message (this vLLM build's reasoning-parser output; clients that expect `reasoning_content` must map it)).

The 95K "worker kill" recorded above for Flash-Next was a property of the August recipe, not of the model: on the cluster recipe the 200K tier of our eval bank ran two full rounds with zero crashes, and at 1M the needle probe recovered the planted content 9/9 at each of 400K, 700K and 950K (three 950K cold prefills of 856-860 s, all correct).

**Why.** With both models in thinking mode on our private 11-category eval bank (questions not published; the full 11-row table, gates and method are in the agentic-thinking cookbook linked below), Qwen3.8-Flash-Next passed all 11 category gates:

| Metric (same bank, same day, thinking mode on both sides) | Qwen3.8-Flash-Next | DeepSeek V4 Flash Vision-Exp |
|---|---|---|
| own-mean (mean of the 8 non-pack categories, percentage points) | 87.9 (+3.8) | 84.1 |
| c8-judgment | 98.3 | 81.7 |
| c9-long-coding | 100.0 | 82.3 |
| c7-agentic-if (median of 5 runs: 90.0 / 93.3 / 88.3 / 91.7 / 91.7) | 91.7 | 90.0 |
| single-stream decode, 400-token prose completion (tok/s) | 51.2 | 32.8 |
| six-stream aggregate (tok/s) | 87.6 | 82.8 |
| KV pool at `--max-model-len` 1M (tokens) | 3,455,574 | 1,492,180 |

The 1,492,180-token KV pool is the DeepSeek stack as it ran after the 2026-09-17 stack switch described in the previous update; the 2.33M figure earlier in this README belongs to the older image.

In **non-thinking** mode (both models greedy, same bank, three-run median) Qwen3.8-Flash-Next had lost the agentic-if category, 81.7 vs 95.0, which is why the switch waited for the thinking-mode rerun rather than shipping on the non-thinking numbers.

**Status as of 2026-10: this is no longer a rollback target.** The mode that started this exact stack was removed when we deleted its image from both nodes on 2026-09-24. The launch command above is still documented but has not been re-run since; the platform has changed (see Update 2026-10). It may still suit **text-only** workloads that do not rely on prefix-cache hits; re-validate it on your own platform first.

**Details in four new cookbooks** (repos under github.com/ryangu00/):

- `dell-pro-max-gb10-qwen3.8-flash-next-1m-context` — YaRN factor 4 to 1M: recipe, KV pool, needle recall, cold-prefill timings
- `dell-pro-max-gb10-vllm-mtp-async-runaway` — the MTP-4 × async-scheduling runaway loop, its single-variable isolation (3/120 → 0/120) and the `--no-async-scheduling` fix
- `dell-pro-max-gb10-qwen3.8-flash-next-agentic-thinking` — why the agentic-if gap was a measurement setting, with the full thinking-mode table
- `dell-pro-max-gb10-qwen3.8-flash-next-engine-ab` — engine-form A/B: native 262K vs YaRN 2 vs YaRN 4 (1M) vs KV bf16 vs MTP off, each a single-variable change
