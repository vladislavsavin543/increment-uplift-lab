"""Сборка автономного демо для GitHub Pages без клиентских данных."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"
REPOSITORY = "https://github.com/vladislavsavin543/increment-uplift-lab"


def main():
    dashboard = (ROOT / "reports" / "dashboard.html").read_text(encoding="utf-8")
    if "__RESULTS__" in dashboard:
        raise ValueError("Сначала пересоберите дашборд: python run.py")
    dashboard = dashboard.replace(
        'href="REPORT_RU.md"',
        f'href="{REPOSITORY}/blob/main/reports/REPORT_RU.md"',
    ).replace(
        'href="tables/test_policies.csv"',
        f'href="{REPOSITORY}/blob/main/reports/tables/test_policies.csv"',
    )
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(dashboard, encoding="utf-8")
    (SITE / ".nojekyll").touch()
    print("Демо собрано: _site/index.html. Клиентские данные не копируются.")


if __name__ == "__main__":
    main()
