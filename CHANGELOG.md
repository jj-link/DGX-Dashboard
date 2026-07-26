# Changelog

### [Unreleased]

- Feature: consolidate workstation and DGX Spark serving, benchmarking, monitoring, and durable web operations into one authenticated control plane.
- Change: import the former inference repository history and runtime control tree without moving model weights, caches, images, or generated results into Git.
- Change: centralize benchmark results, corpus data, run metadata, logs, and the incremental result index under `/var/lib/dgx-dashboard`.
- Security: require private binding, Basic authentication, same-origin JSON mutations, strict request limits, validated recipes, exact resource leases, and labeled cleanup.
- Fix: proxy the WSL loopback dashboard through Windows Tailscale HTTPS so local and remote tailnet clients use the same reachable origin.
- Fix: preserve API responsiveness during long serving verification and prevent overlapping live-stat polling.
- Fix: reject incomplete Hugging Face model and drafter snapshots before single-node or cluster launch.
- Fix: preserve durable run history when a serving recipe leaves the active catalog, without allowing that retired recipe to launch.
- Fix: keep the last successful serving recipe visible after a failed start, reject occupied target ports before Docker creation, and guide operators to stop the active recipe first.
- Change: verify start, model response, stop, and cleanup through the canonical repository on `local`, all three Spark nodes, and the Spark 2 + Spark 3 cluster.
