#!/usr/bin/env bash
# Serve benchmark-dashboard.html on port 8080
cd ~/inference/benchmarks/results
python3 -m http.server 8080
