#!/usr/bin/env bash
# Check DSpark cluster status
set -eu

echo "=== API Health ==="
if curl -fsS --max-time 5 "http://100.92.139.82:8888/v1/models" >/dev/null 2>&1; then
  echo "API: UP (http://100.92.139.82:8888/v1)"
  curl -s --max-time 5 "http://100.92.139.82:8888/v1/models" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f'  Model: {d[\"data\"][0][\"id\"]}')" 2>/dev/null
else
  echo "API: DOWN"
fi

echo ""
echo "=== spark2-ts (rank 0) ==="
ssh spark2-ts "docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null || echo '  cannot reach spark2-ts'"

echo ""
echo "=== spark3-ts (rank 1) ==="
ssh spark3-ts "docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null || echo '  cannot reach spark3-ts'"
