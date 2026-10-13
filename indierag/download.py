"""Dataset downloader (idempotent, resumable). Honors HF_ENDPOINT (e.g. https://hf-mirror.com)."""
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

FILES = {
    # TriviaQA rc.web validation: questions + real Bing search-result pages (url, title, full text)
    **{f"triviaqa/rc.web/validation-0000{i}-of-00003.parquet":
       f"{HF}/datasets/mandarjoshi/trivia_qa/resolve/main/rc.web/validation-0000{i}-of-00003.parquet" for i in range(3)},
    # HotpotQA distractor dev (multi-hop agent experiments)
    "hotpotqa/distractor/validation-00000-of-00001.parquet":
        f"{HF}/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet",
    # RAMDocs (Wang et al. 2025): ambiguity + misinformation + noise, realistic conflicting evidence
    "ramdocs/RAMDocs_test.jsonl": "https://raw.githubusercontent.com/HanNight/RAMDocs/main/RAMDocs_test.jsonl",
}


def fetch(url, dst: Path, retries=5):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    part = Path(str(dst) + ".part")
    for attempt in range(retries):
        try:
            pos = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **({"Range": f"bytes={pos}-"} if pos else {})})
            if os.environ.get("HF_TOKEN") and "huggingface" in url:
                req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
            with urllib.request.urlopen(req, timeout=120) as r:
                if pos and r.status != 206:
                    pos = 0
                    part.unlink(missing_ok=True)
                with open(part, "ab") as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
            part.rename(dst)
            print(f"[download] ok {dst}", flush=True)
            return dst
        except Exception as e:  # noqa
            print(f"[download] retry {attempt + 1}/{retries} {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def main():
    for rel, url in FILES.items():
        fetch(url, RAW / rel)


if __name__ == "__main__":
    main()
