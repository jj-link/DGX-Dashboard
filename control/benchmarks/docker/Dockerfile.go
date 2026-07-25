# polybench-go: every exercism Go exercise is stdlib-only (no external
# modules in any go.mod), so the base toolchain already runs fully offline.
# Pinned here only for reproducibility / consistent tagging.
FROM golang:1.22
ENV GOFLAGS=-mod=mod
