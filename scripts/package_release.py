#!/usr/bin/env python3
"""Package the foreground-coverage project and verify every archived file."""
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent


def main():
    excluded = {"__pycache__", ".git", ".pytest_cache"}
    files = sorted(p for p in PROJECT.rglob("*") if p.is_file()
                   and not excluded.intersection(p.relative_to(PROJECT).parts)
                   and p.name not in {"PACKAGE_MANIFEST.json", "release_validation.json"}
                   and not set(p.relative_to(PROJECT).parts).intersection({".git", "work", "dist", ".venv"}))
    records = [{"path": str(p.relative_to(PROJECT)), "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    manifest = PROJECT / "PACKAGE_MANIFEST.json"
    manifest.write_text(json.dumps({"version": 11, "files": records,
        "contents": "Foreground-consistent bidirectional coverage implementation, baseline comparison, results, report and example bank only.",
        "not_bundled": "Raw datasets, model weights and feature caches."}, indent=2) + "\n")
    dist = PROJECT / "dist"
    dist.mkdir(exist_ok=True)
    archive = dist / "anomalydino-foreground-coverage.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files + [manifest]:
            z.write(p, str(p.relative_to(PROJECT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in records:
            assert hashlib.sha256(z.read(r["path"])).hexdigest() == r["sha256"]
        assert not any("/experiments/" in name for name in z.namelist())
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(sha + "  " + archive.name + "\n")
    result = {"status": "pass", "version": 11, "files": len(files) + 1,
              "archive_bytes": archive.stat().st_size, "sha256": sha,
              "crc_and_all_file_hashes_verified": True}
    (PROJECT / "results/release_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
