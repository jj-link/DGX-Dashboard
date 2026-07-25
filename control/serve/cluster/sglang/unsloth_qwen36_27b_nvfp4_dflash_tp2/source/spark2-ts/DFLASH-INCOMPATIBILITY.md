# DFlash TP=2 investigation result

Final result: SGLang + DFlash + TP=2 across Spark2/Spark3 works when CUDA graphs are disabled.

The initial Spark1-baseline-preserving candidate used `--cuda-graph-max-bs 16` and stalled on long generation. That was not ultimately a DFlash incompatibility; the failure reproduced in the non-DFlash candidate too, so the concrete incompatibility was CUDA graph capture/use on this SGLang/GB10/multi-node path. The final recipe keeps DFlash enabled and sets `--disable-cuda-graph`.

Candidate that failed:

- Image: `lmsysorg/sglang:v0.5.12-cu130`
- Target: `/models/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a`
- Drafter: `/models/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106`
- Spark2 rank0 + Spark3 rank1, `--tp-size 2 --nnodes 2 --dist-init-addr 10.0.0.1:25001`
- Preserved DFlash flags: `DFLASH`, draft tokens `20`, draft window `4096`, `extra_buffer`
- Preserved Spark1 CUDA graph flag: `--cuda-graph-max-bs 16`

Observed failure with CUDA graphs enabled:

- `/v1/models` worked.
- Short chat worked.
- Logs confirmed DFlash initialized and was actually used: `Initialized DFLASH draft runner`, `DFLASH draft runner ready`, `DFLASH verify completed`.
- Long generation with ~19k prompt tokens reached the end of chunked prefill, then stalled and did not return before the 600s verifier timeout.
- The same class of stall reproduced without DFlash after some decode batches, with one node active and the other idle.

Fix applied in final recipe:

- Keep DFlash enabled.
- Replace the Spark1 CUDA graph setting with `--disable-cuda-graph` for multi-node GB10.

Final verification with DFlash + `--disable-cuda-graph`:

- `/v1/models`: HTTP 200.
- Short chat: HTTP 200.
- Long generation: HTTP 200, 1024 completion tokens, elapsed 49.13s.
- Both GPUs active during long generation: Spark2 avg util 88.7%, Spark3 avg util 86.0%.
- Logs confirmed RoCE/IB: `NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE` and rank-to-rank channels over `NET/IB/0`.
- Logs confirmed DFlash path: accept length/rate entries such as `accept len: 2.77, accept rate: 0.09`, plus DFlash init messages.
- Last 10 logged generation throughput samples averaged 22.41 tok/s; latest sample was 27.23 tok/s.
