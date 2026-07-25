#!/usr/bin/env bash
# Check MiMo-V2.5-NVFP4 cluster status
set -eu

echo "=== API Health ==="
if curl -fsS --max-time 5 "http://100.92.139.82:8888/v1/models" >/dev/null 2>&1; then
  echo "API: UP (http://100.92.139.82:8888/v1)"
  curl -s --max-time 5 "http://100.92.139.82:8888/v1/models" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f'  Model: {d[\"data\"][0][\"id\"]}'); print(f'  Max context: {d[\"data\"][0][\"max_model_len\"]}')" 2>/dev/null
else
  echo "API: DOWN"
fi

echo ""
echo "=== spark2-ts (head, rank 0) ==="
ssh spark2-ts "docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null | grep mimo || echo '  no MiMo container'" 2>/dev/null || echo "  cannot reach spark2-ts"

echo ""
echo "=== spark3-ts (worker, rank 1) ==="
ssh spark3-ts "docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null | grep mimo || echo '  no MiMo container'" 2>/dev/null || echo "  cannot reach spark3-ts"
