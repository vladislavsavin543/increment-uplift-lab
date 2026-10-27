"""Архив исходного кода и результатов без данных на уровне клиентов."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    destination=ROOT.parent/"increment-uplift-lab.zip"
    required=["README.md","START_HERE_RU.md","INTERVIEW_RU.md","run.py",
              "reports/dashboard.html","reports/REPORT_RU.md","reports/results.json",
              "notebooks/01_uplift_walkthrough.ipynb","reports/figures/calculator_preview.png"]
    for name in required:
        if not (ROOT/name).is_file():
            raise FileNotFoundError(name)
    paths=[]
    for path in sorted(ROOT.rglob("*")):
        rel=path.relative_to(ROOT)
        if not path.is_file() or any(part in {".git","__pycache__",".pytest_cache",".venv"} for part in rel.parts):
            continue
        if rel.parts[:2]==("data","processed"):
            continue
        if rel.parts[:2]==("data","raw") and path.name!="manifest.json":
            continue
        paths.append(path)
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path,Path(ROOT.name)/path.relative_to(ROOT))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Проверка целостности архива не пройдена")
    print(json.dumps(dict(archive=destination.name,files=len(paths),bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),indent=2))

if __name__=="__main__":
    main()

