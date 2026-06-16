import os
import urllib.request
from pathlib import Path

# URLs provided for the model weights
INTENT_URL = "https://storage.googleapis.com/explanability-for-temporal-graphs/intent/model.safetensors"
PARSER_URL = "https://storage.googleapis.com/explanability-for-temporal-graphs/parser/model.safetensors"

# Destination paths
ROOT = Path(__file__).resolve().parents[1]
INTENT_DIR = ROOT / "experiments" / "m2_e3_parse" / "artifacts" / "intent"
PARSER_DIR = ROOT / "experiments" / "m2_e3_parse" / "artifacts" / "parser"

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {dest}...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"Successfully downloaded {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    print("Downloading Milestone 2 model weights...")
    download_file(INTENT_URL, INTENT_DIR / "model.safetensors")
    download_file(PARSER_URL, PARSER_DIR / "model.safetensors")
    print("\nDownload complete. Models are ready for inference.")

if __name__ == "__main__":
    main()
