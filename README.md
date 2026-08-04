# DGX Dashboard Unified Control Plane

One workstation-hosted dashboard for GPU monitoring, model lifecycle operations, and oneshot benchmark runs across:

- `local` — RTX PRO 6000 Blackwell workstation
- `spark1`, `spark2`, `spark3` — individual DGX Spark nodes
- `cluster` — the Spark 2 + Spark 3 two-node target

The dashboard keeps source and control logic in this repository. Model weights, Hugging Face caches, Docker images and volumes, run state, logs, benchmark corpora, and raw results remain external runtime data.

## Runtime layout

| Purpose | Path |
|---|---|
| Canonical checkout | `/home/workbench/Projects/personal/dgx-dashboard` |
| Production config | `/var/lib/dgx-dashboard/config-production.ini` |
| Auth environment | `/var/lib/dgx-dashboard/dashboard.env` (`0600`) |
| Durable runs and logs | `/var/lib/dgx-dashboard/runs` |
| Oneshot results | `/var/lib/dgx-dashboard/benchmark-results` |
| Aider results | `/var/lib/dgx-dashboard/aider-benchmarks` |
| Polyglot corpus | `/var/lib/dgx-dashboard/polyglot-benchmark` |
| Result index | `/var/lib/dgx-dashboard/result-index.json` |

`config.ini` is the safe, tracked example and keeps `[control] enabled = false`. Enable controls only in an external production config after all preflights pass.

## Requirements

- Python 3.10 or newer
- Docker access for the dashboard service user
- NVIDIA userspace tools on the workstation
- Passwordless, host-key-pinned SSH aliases `spark1-ts`, `spark2-ts`, and `spark3-ts`
- Tailscale connectivity between the workstation and Spark nodes
- Existing model/cache roots referenced by recipe `runtime.env` files

Install Python dependencies:

```bash
cd /home/workbench/Projects/personal/dgx-dashboard
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Configuration and authentication

Create a production config outside Git, using `config.ini` as the schema. Bind Waitress to WSL loopback, set the canonical wrapper and state roots, set the exact Windows Tailscale Serve origin, enable every target, and then enable controls:

```ini
[server]
host = 127.0.0.1
port = 9000
refresh_interval = 3

[control]
enabled = true
allowed_origin = https://jjlink-pc-1.tail90c6fe.ts.net:8443
wrapper_root = /home/workbench/Projects/personal/dgx-dashboard
state_dir = /var/lib/dgx-dashboard/runs
polyglot_root = /var/lib/dgx-dashboard/polyglot-benchmark
targets = local,spark1,spark2,spark3,cluster
retention = 250
serving_timeout = 1800
benchmark_timeout = 86400
```

Store Basic-auth credentials only in the service environment file:

```bash
install -m 0600 /dev/null /var/lib/dgx-dashboard/dashboard.env
# Add DASHBOARD_AUTH_USER and DASHBOARD_AUTH_PASSWORD without committing them.
```

A control-enabled startup fails closed if auth, origin, Docker, GPU, SSH, Git cleanliness, state roots, result roots, corpus, or recipe validation fails. Mutation requests additionally require Basic auth, `Content-Type: application/json`, the configured same-origin `Origin`, and a body no larger than 16 KiB. No permissive CORS headers are emitted.

Docker-group membership is effectively host-root capability. Keep Waitress bound to WSL loopback and expose it only through the Windows Tailscale HTTPS proxy. In Windows PowerShell:

```powershell
tailscale serve --bg --https=8443 --yes http://127.0.0.1:9000
tailscale serve status
```

This publishes the dashboard only inside the tailnet at `https://jjlink-pc-1.tail90c6fe.ts.net:8443/`. Local Windows access and remote tailnet access use the same authenticated HTTPS origin.

## System service

`dashboard.service` is the canonical system unit. It runs as `workbench`, uses the external production config and auth environment, keeps the full process group under systemd, and restricts writes to `/var/lib/dgx-dashboard`.

```bash
sudo install -m 0644 dashboard.service /etc/systemd/system/dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable --now dashboard.service
sudo systemctl status dashboard.service --no-pager
```

Smoke checks:

```bash
curl -u "$DASHBOARD_AUTH_USER:$DASHBOARD_AUTH_PASSWORD" \
  https://jjlink-pc-1.tail90c6fe.ts.net:8443/api/control/catalog
curl -u "$DASHBOARD_AUTH_USER:$DASHBOARD_AUTH_PASSWORD" \
  https://jjlink-pc-1.tail90c6fe.ts.net:8443/api/stats
```

Open `https://jjlink-pc-1.tail90c6fe.ts.net:8443/` and authenticate. The **Live** tab monitors GPUs and inference endpoints, **Benchmarks** browses indexed results, and **Control** exposes typed operations, validated launch-profile choices, bounded logs, history, cancellation, and result links.

## Model lifecycle CLI

List the validated catalog:

```bash
./serve.sh --help
```

Start a recipe. Extra tokens after a single-device artifact are passed only to the selected engine launcher. Cluster recipes use typed lifecycle arguments:

```bash
./serve.sh <local|spark1|spark2|spark3> <vllm|sglang> <artifact> [engine args...]
./serve.sh cluster <vllm|sglang> <artifact> [start|status|logs|verify|stop] [profile] [action args...]
```

Read or mutate an exact recipe-owned service:

```bash
./serve.sh <target> <engine> <artifact> status
./serve.sh <target> <engine> <artifact> logs [1-1000]
./serve.sh <target> <engine> <artifact> verify
./serve.sh <target> <engine> <artifact> stop
```

Single-device services bind the workstation model to loopback and Spark models to each node's Tailscale address. The two-node cluster binds its head API to Spark 2's Tailscale address on port `8888`; Spark 3 remains headless. Cluster lifecycle actions operate on both exact rank containers and verify two-way tensor parallelism plus RoCE/NCCL topology.

The DeepSeek V4 Flash DSpark cluster recipe has three validated launch profiles:

| Profile | KV cache | Context | Sequences | Speculation |
|---|---|---:|---:|---:|
| `quality` (default) | FP8 DS-MLA | 1,048,576 | 6 | MTP3 |
| `balanced` | NVFP4 DS-MLA | 1,048,576 | 6 | MTP3 |
| `throughput` | NVFP4 DS-MLA | 350,000 | 12 | MTP5 |

Pass the profile after every lifecycle action; omitting it selects the tracked `quality` default:

```bash
artifact=deepseek_ai_deepseek_v4_flash_dspark_tp2
./serve.sh cluster vllm "$artifact" start throughput
./serve.sh cluster vllm "$artifact" status throughput
./serve.sh cluster vllm "$artifact" logs throughput 250
./serve.sh cluster vllm "$artifact" verify throughput
./serve.sh cluster vllm "$artifact" stop throughput
```

Artifact plus launch profile is the exact lifecycle identity. Status probes all three profiles independently, even for containers started outside the dashboard. A split-rank or multiple-profile condition is reported as a conflict and blocks lifecycle mutations. In the **Control** tab, selecting this DeepSeek artifact reveals the profile dropdown and its KV-cache, context, sequence, and speculation settings.

Starts never replace an occupied target implicitly. Stop the current exact recipe first. Container, image, network, port, model, and recipe identity checks fail closed rather than touching an unknown service.

Before a cached repository is mounted, the launcher rejects broken snapshot symlinks and files missing from a Hugging Face weight index. An interrupted download cannot be treated as an installed model or drafter on either single-node or cluster paths.

## Oneshot benchmark CLI

Benchmarks always run on the workstation against the model already serving on the selected target:

```bash
./benchmark.sh <local|spark1|spark2|spark3|cluster> oneshot [options]
```

The CLI discovers the served model from `/v1/models`; there is no separate alias argument. Omitting `--lang` runs `cpp`, `go`, `java`, `javascript`, `python`, and `rust`. Common options include:

```text
--lang {cpp,go,java,javascript,python,rust}
--num-tests N
--keywords NAME,...
--max-tokens N
--temperature FLOAT
--timeout SEC
--test-timeout SEC
--concurrency N
--reasoning VALUE
--reasoning-effort {high,max}
--engine-version VALUE
--runtime-image VALUE
--runtime-image-digest VALUE
--model-source VALUE
--model-revision VALUE
```

Each result records target and run metadata and uses a collision-resistant filename. Grading containers carry exact `io.dgx-dashboard.kind=benchmark` and `io.dgx-dashboard.run-id=<uuid>` labels. Only those labels are used for cancellation, timeout, and restart cleanup.

## HTTP control API

Read routes:

- `GET /`
- `GET /api/stats`
- `GET /api/benchmarks`
- `GET /api/serving`
- `GET /api/control/catalog`
- `GET /api/serving/<target>/<engine>/<artifact>/logs?launch_profile=<name>&lines=N`
- `GET /api/runs?limit=N`
- `GET /api/runs/<uuid>`
- `GET /api/runs/<uuid>/log?offset=N&limit=N`
- `GET /api/benchmarks/results/<relative-path>`

Mutation routes:

- `POST /api/runs` — typed serving or benchmark request
- `POST /api/runs/<uuid>/cancel`

Run metadata never exposes credentials, PIDs, argv, or environments. A profiled serving request must include its validated `launch_profile`; unprofiled recipes use `null`. Logs use capped byte-cursor reads. Metadata is persisted atomically; on restart, nonterminal serving runs are reconciled against exact artifact-and-profile service state, benchmark runs become `interrupted`, exact labeled benchmark containers are removed, and unrelated containers are left untouched. Terminal history remains readable if a serving recipe is later removed from the active catalog; historical DeepSeek records without a profile retain the former `quality` behavior.

Resource leases prevent overlapping mutations on the same target. `cluster` conflicts with individual `spark2` and `spark3` mutations. Only one benchmark worker may run at a time.

Only completed, unsampled runs covering all six languages are merged into the Benchmarks tab's comparative Oneshot table. Single-language, `--num-tests`, and keyword-filtered runs remain available through run-history result links but are excluded from that table. Each dynamic row shows its target, start timestamp, exact reasoning effort, weight quantization, and short run ID; reopening the tab fetches current results instead of retaining an earlier page-local snapshot.

Control startup validates global authentication, binding, repository, and durable-storage invariants without requiring every target to be online. Each submitted mutation then checks only its own infrastructure: local operations require workstation Docker and the expected GPU, remote operations require SSH to that Spark, and `cluster` requires both Spark 2 and Spark 3; benchmarks also require workstation Docker. An unavailable dependency returns HTTP `503` with code `target_unavailable` before a run ID, resource lease, or durable record is created. Healthy targets remain operable while another target is offline.

## Model capability discovery

Serving recipes publish client-relevant metadata from the live inference endpoint at `GET /v1/model-capabilities`. The versioned document identifies the active served model, context and output limits, weight quantization, input and tool support, and the exact reasoning levels and request format. Capability-backed servers fail startup when the profile is missing or malformed; cluster lifecycle verification also rejects profile drift.

The reusable OMP integration under `control/clients/omp/` reads this endpoint during startup and registers the currently served model with those capabilities. `spark-cluster.ts` covers the Spark 2 + Spark 3 vLLM service, while `local-sglang.ts` covers the RTX 6000 SGLang service. The SGLang launcher attaches the capability middleware directly to SGLang's FastAPI application rather than adding another inference proxy. OMP therefore does not require per-model `modelOverrides` entries when either service changes to another capability-backed recipe; the serving recipe remains the authority for runtime behavior and client metadata.

## Repository synchronization

The workstation is the source controller. Synchronize clean, published commits to Spark checkouts:

```bash
./sync.sh spark1-ts
./sync.sh spark2-ts
./sync.sh spark3-ts
./sync.sh all
```

Synchronization rejects dirty, detached, divergent, or unpublished states. It never pulls model weights, results, caches, or run state into Git.

## Verification

Run both suites from the canonical checkout:

```bash
venv/bin/python -m pytest -q
./control/tests/run.sh
```

Useful focused smoke checks:

```bash
./serve.sh <target> <engine> <artifact> status
./serve.sh <target> <engine> <artifact> verify
curl --fail http://<target-endpoint>/v1/models
curl --fail http://<target-endpoint>/v1/model-capabilities
```

For a serving cutover, capture a deterministic chat response before stop, start the same recipe from this repository, repeat the same request, and compare model, content, and finish reason.

## Rollback backup policy

The original `inference` checkout is retained unchanged as a rollback backup, not as an active source or runtime caller. This migration does not archive its GitHub repository. Its Git history remains reachable from this repository through the imported source-history merge. Runtime results and corpus data stay under `/var/lib/dgx-dashboard`; rollback symlinks expose those same trees to old tools without copying them.

Passing cutover checks does not authorize deletion. Do not delete or modify the backup checkout, Spark 1 legacy-dashboard backup, migration manifests, or rollback links unless the operator explicitly approves a separate removal action.
