import io
import json
import tarfile
from pathlib import Path

BLOB_ROOT = Path("/containerd/io.containerd.content.v1.content/blobs/sha256")
OUTPUT = Path("/out/original-hybrid-image.tar")
INDEX_DIGEST = "302d66e7db9206f8f606c575d4603d56502d97bf5214319bbabd4514b736c0a5"
TAG = "vllm:v0.24.0-dflash-hybrid-20260716"


def digest_path(digest: str) -> Path:
    algorithm, value = digest.split(":", 1)
    if algorithm != "sha256":
        raise ValueError(f"unsupported digest algorithm: {algorithm}")
    path = BLOB_ROOT / value
    if not path.is_file():
        raise FileNotFoundError(f"missing containerd blob: {digest}")
    return path


def collect_descriptor(descriptor: dict, collected: dict[str, Path]) -> None:
    digest = descriptor["digest"]
    if digest in collected:
        return
    path = digest_path(digest)
    collected[digest] = path
    media_type = descriptor.get("mediaType", "")
    if "image.index" not in media_type and "image.manifest" not in media_type:
        return
    document = json.loads(path.read_text())
    for child in document.get("manifests", []):
        collect_descriptor(child, collected)
    config = document.get("config")
    if config:
        collect_descriptor(config, collected)
    for child in document.get("layers", []):
        collect_descriptor(child, collected)


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


index = json.loads((BLOB_ROOT / INDEX_DIGEST).read_text())
index["manifests"] = index["manifests"][:1]
index["manifests"][0].setdefault("annotations", {})[
    "org.opencontainers.image.ref.name"
] = TAG
collected: dict[str, Path] = {}
for manifest in index["manifests"]:
    collect_descriptor(manifest, collected)

total_bytes = sum(path.stat().st_size for path in collected.values())
print(
    f"restoring {len(collected)} blobs ({total_bytes} bytes) as {TAG}",
    flush=True,
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(OUTPUT, "w") as archive:
    add_bytes(archive, "oci-layout", b'{"imageLayoutVersion":"1.0.0"}\n')
    add_bytes(
        archive,
        "index.json",
        json.dumps(index, separators=(",", ":")).encode() + b"\n",
    )
    for digest, path in collected.items():
        _, value = digest.split(":", 1)
        archive.add(path, arcname=f"blobs/sha256/{value}", recursive=False)
print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)", flush=True)
