"""Download the data attempt 14 reads from the two GitHub releases and unpack it into data/.

Run from the repository root:  python scripts/download_data.py
(it uses the standard library only, so any Python 3.10+ works)

Two parts, each a zip with a manifest of sha256 checksums:
  1. final_study_data.zip (79 MB, 53 files): the meeting panels, corpus, labels and economic vintages, from the release
     data-2026-10-18 of the original repository (fed-news-forecasting-final); manifest MANIFEST.txt.
  2. split_tone_data.zip (40 MB, 7 files): the Haiku teacher probabilities, the distilled reader's sentence
     scores and document tones, and the two other readers' document tones that fix the common meetings, from the
     release data-2026-10-31 of this repository; manifest MANIFEST_SPLIT_TONE.txt.
Every file is checked against its manifest. config.py then finds data/ automatically. Run it again to re-check; a
part whose files are already correct is not downloaded again. docs/DATA.md describes every file.
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PARTS = [  # (zip url, manifest name inside the zip)
    ("https://github.com/taminggg1-cmyk/fed-news-forecasting-final/releases/download/data-2026-10-18/"
     "final_study_data.zip", "MANIFEST.txt"),
    ("https://github.com/taminggg1-cmyk/distilled-tone-split-doctype/releases/download/data-2026-10-31/"
     "split_tone_data.zip", "MANIFEST_SPLIT_TONE.txt"),
]
TOP = "fed_news_data/"  # folder name inside both zips


def sha256(path: Path) -> str:
    """sha256 of a file, read in 1 MB pieces."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(dest: Path, name: str) -> dict[str, str]:
    """{relative path: sha256} from the manifest dest/name (first line is a header)."""
    rows = (dest / name).read_text(encoding="utf-8").splitlines()[1:]
    return {line.split()[2]: line.split()[0] for line in rows if line.strip()}


def check(dest: Path, name: str) -> list[str]:
    """Files listed in the manifest `name` that are missing or whose checksum differs."""
    if not (dest / name).exists():
        return [name]
    return [rel for rel, digest in manifest(dest, name).items()
            if not (dest / rel).exists() or sha256(dest / rel) != digest]


def download(url: str, target: Path) -> None:
    """Download url to target, printing progress."""
    with urllib.request.urlopen(url) as resp, open(target, "wb") as out:
        total, done = int(resp.headers.get("Content-Length", 0)), 0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if total:
                print(f"\rdownloading {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def unpack(zip_path: Path, dest: Path) -> None:
    """Extract every file under fed_news_data/ in the zip into dest (never outside it)."""
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(TOP):
                continue
            out = (dest / info.filename[len(TOP):]).resolve()
            if root not in out.parents:
                raise ValueError(f"unsafe path in zip: {info.filename}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main(argv=None) -> int:
    """Download and unpack each part unless data/ already matches its manifest; return 0 when every file checks out."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "data"), help="target folder")
    ap.add_argument("--zip-dir", help="a folder holding already downloaded zips, used instead of downloading")
    args = ap.parse_args(argv)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for url, name in PARTS:
        if not check(dest, name):
            print(f"{name}: all files present and correct")
            continue
        zip_name = url.rsplit("/", 1)[1]
        zip_path = Path(args.zip_dir) / zip_name if args.zip_dir else dest / zip_name
        if not args.zip_dir:
            download(url, zip_path)
        unpack(zip_path, dest)
        if not args.zip_dir:
            zip_path.unlink()
        bad = check(dest, name)
        if bad:
            print(f"{name}: checksum mismatch or missing: {bad}")
            return 1
        print(f"{name}: {len(manifest(dest, name))} files unpacked and checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
