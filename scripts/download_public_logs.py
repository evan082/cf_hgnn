"""Download public logs via HuggingFace / alternate mirrors (4TU often 403)."""
from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public_logs"
DATA.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# (name, out_xes, urls) — urls tried in order
JOBS = [
    (
        "sepsis",
        "sepsis.xes",
        [
            "https://huggingface.co/datasets/kondratevakate/sepsis-cases-event-log/resolve/main/sepsis.xes.gz",
        ],
    ),
    (
        "traffic",
        "traffic.xes",
        [
            # same-origin 4TU file URL (figshare ndownloader often WAF 202)
            "https://data.4tu.nl/file/806acd1a-2bf2-4e39-be21-69b8cad10909/b234b06c-4d4f-4055-9f14-6218e3906d82",
            "https://ndownloader.figshare.com/files/24018146",
        ],
    ),
    (
        "bpi2013",
        "bpi2013_incidents.xes",
        [
            "https://ais.win.tue.nl/bpi/2013/bpi_challenge_2013_incidents.xes.gz",
            "https://data.4tu.nl/file/500573e6-accc-4b0c-9576-aa5468b10cee/BPI_Challenge_2013_incidents.xes.gz",
        ],
    ),
    (
        "bpic2017",
        "bpic2017.xes",
        [
            "https://data.4tu.nl/file/5f3067df-f10b-45da-b98b-86ae4c7a310b/BPI_Challenge_2017.xes.gz",
            "https://ndownloader.figshare.com/files/24015728",
        ],
    ),
]


def get(url: str, dest: Path, timeout: int = 300) -> None:
    print(f"GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"  saved {dest} ({dest.stat().st_size} bytes)", flush=True)


def to_xes(download_path: Path, out_xes: Path) -> None:
    with open(download_path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(download_path, "rb") as fin, open(out_xes, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        download_path.unlink(missing_ok=True)
    else:
        download_path.replace(out_xes)


def main() -> int:
    ok_any = False
    for name, out_name, urls in JOBS:
        out = DATA / out_name
        if out.exists() and out.stat().st_size > 10_000:
            print(f"SKIP {name}: {out}", flush=True)
            ok_any = True
            continue
        tmp = DATA / f"{name}.part"
        done = False
        for url in urls:
            try:
                get(url, tmp)
                to_xes(tmp, out)
                print(f"OK {name}: {out} ({out.stat().st_size})", flush=True)
                done = True
                ok_any = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"  fail: {e}", flush=True)
                tmp.unlink(missing_ok=True)
        if not done:
            print(f"FAILED {name}", flush=True)
    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
