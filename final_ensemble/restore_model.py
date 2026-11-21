"""Restore verified inference weights from the release assets bundled with this Git revision."""
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_member(name):
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or "\\" in name or ":" in name or ".." in parts:
        raise ValueError(f"Invalid archive path: {name}")
    if "/".join(parts) != name:
        raise ValueError(f"Noncanonical archive path: {name}")
    return parts


def verify_tree(root, expected):
    existing = {p.relative_to(root).as_posix() for p in (root / "models").rglob("*") if p.is_file()}
    if set(expected) != existing:
        raise ValueError("The models directory does not match this release. Restore in a clean checkout.")
    for name, signature in expected.items():
        if sha(root / name) != signature:
            raise ValueError("Mismatched model file: " + name)


def restore(root, parts_dir, full_zip=None):
    manifest = json.loads((root / "weights_manifest.json").read_bytes())
    expected = manifest["model_files"]
    for name, digest in expected.items():
        if safe_member(name)[0] != "models" or len(digest) != 64:
            raise ValueError("Invalid model manifest.")
    for name, signature in manifest["inference_code"].items():
        safe_member(name)
        if sha(root / name) != signature:
            raise ValueError("Inference code and release manifest disagree: " + name)
    if (root / "models").exists():
        verify_tree(root, expected)
        print("Verified existing weights.")
        return
    with tempfile.TemporaryDirectory(prefix="restore-", dir=root) as tmp:
        stage = Path(tmp)
        if full_zip is None:
            archive = stage / "inference.zip"
            with archive.open("wb") as destination:
                for part in manifest["parts"]:
                    name = part["name"]
                    if len(safe_member(name)) != 1:
                        raise ValueError("Invalid part name.")
                    path = parts_dir / name
                    if path.stat().st_size != part["bytes"]:
                        raise ValueError("Incomplete release part: " + name)
                    h = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(4 << 20), b""):
                            h.update(block)
                            destination.write(block)
                    if part["sha256"] != h.hexdigest():
                        raise ValueError("Checksum mismatch: " + name)
        else:
            archive = Path(full_zip)
        if archive.stat().st_size != manifest["archive_bytes"] or sha(archive) != manifest["archive_sha256"]:
            raise ValueError("Inference archive checksum mismatch.")
        with zipfile.ZipFile(archive) as bundle:
            if len(bundle.namelist()) != len(set(bundle.namelist())):
                raise ValueError("Duplicate ZIP member.")
            for name in expected:
                dest = stage.joinpath(*safe_member(name))
                dest.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as src, dest.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=4 << 20)
        verify_tree(stage, expected)
        (stage / "models").rename(root / "models")
    print("Verified and installed inference weights.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("release_assets"))
    parser.add_argument("--archive", type=Path, help="Use the complete inference ZIP instead of parts.")
    args = parser.parse_args()
    restore(Path(__file__).resolve().parent, args.assets, args.archive)
