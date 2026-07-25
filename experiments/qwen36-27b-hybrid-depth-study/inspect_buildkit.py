import json
from pathlib import Path

blob_root = Path("/containerd/io.containerd.content.v1.content/blobs/sha256")
for digest in [
    "302d66e7db9206f8f606c575d4603d56502d97bf5214319bbabd4514b736c0a5",
    "4b9bc487ddeb23fcc870a7a95e7868a785f0c8c09466268c83ca2ef1ed5be3e5",
    "26007482b41dc807ae7def45f43f154b009ca61e82893ff092a28e977d38c709",
]:
    path = blob_root / digest
    print(f"{digest}: exists={path.exists()} size={path.stat().st_size if path.exists() else None}")
    if path.exists():
        try:
            print(json.dumps(json.loads(path.read_text()), indent=2))
        except Exception:
            pass

snapshot_root = Path("/containerd/io.containerd.snapshotter.v1.overlayfs")
print(f"{snapshot_root}: exists={snapshot_root.exists()}")
if snapshot_root.is_dir():
    for child in sorted(snapshot_root.iterdir()):
        print(f"  {child.name} directory={child.is_dir()} size={child.stat().st_size}")
