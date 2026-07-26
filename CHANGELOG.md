# Changelog

### [Unreleased]

- Feature: consolidate workstation and DGX Spark serving, benchmarking, monitoring, and durable web operations into one authenticated control plane.
- Change: import the former inference repository history and runtime control tree without moving model weights, caches, images, or generated results into Git.
- Change: centralize benchmark results, corpus data, run metadata, logs, and the incremental result index under `/var/lib/dgx-dashboard`.
- Security: require private binding, Basic authentication, same-origin JSON mutations, strict request limits, validated recipes, exact resource leases, and labeled cleanup.
- Fix: preserve API responsiveness during long serving verification and prevent overlapping live-stat polling.
- Fix: reject incomplete Hugging Face model and drafter snapshots before single-node or cluster launch.
- Fix: preserve durable run history when a serving recipe leaves the active catalog, without allowing that retired recipe to launch.
- Change: verify start, model response, stop, and cleanup through the canonical repository on `local`, all three Spark nodes, and the Spark 2 + Spark 3 cluster.
