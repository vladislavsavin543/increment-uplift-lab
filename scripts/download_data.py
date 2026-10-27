"""Публичное зеркало Hillstrom из scikit-uplift; проверка контрольной суммы."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
ROOT = Path(__file__).resolve().parents[1]
URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
EXPECTED_MD5 = "a68a81291f53a14f4e29002629803ba3"

def main():
    folder = ROOT/"data"/"raw"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"hillstrom.csv.gz"
    if not path.exists():
        with urlopen(URL, timeout=60) as response:
            payload = response.read()
        if hashlib.md5(payload).hexdigest() != EXPECTED_MD5:
            raise ValueError("Контрольная сумма загруженного файла не совпадает")
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.md5(payload).hexdigest() != EXPECTED_MD5:
        raise ValueError("Контрольная сумма не совпадает; восстановите исходные данные")
    manifest = dict(url=URL, md5=EXPECTED_MD5, sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload), source="Kevin Hillstrom / MineThatData (2008)",
        original="https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html",
        mirror_reference="https://www.uplift-modeling.com/en/latest/_modules/sklift/datasets/datasets.html#fetch_hillstrom")
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Данные проверены: {path.name}, {len(payload):,} байт")

if __name__ == "__main__":
    main()

