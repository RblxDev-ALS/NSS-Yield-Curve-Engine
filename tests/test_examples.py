import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "tour.ipynb"


def test_tour_notebook_is_valid_python():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 5
    for cell in code:
        src = "".join(line for line in cell["source"] if not line.lstrip().startswith("%"))
        ast.parse(src)
        assert not cell["outputs"], "commit the notebook without outputs"


def test_tour_notebook_uses_existing_api():
    import nss_engine.analytics as analytics
    import nss_engine.regime as regime
    import nss_engine.termpremium as termpremium

    for mod, name in [
        (termpremium, "fit_acm"),
        (termpremium, "zero_panel"),
        (regime, "recession_probability_model"),
        (analytics, "risk_report"),
    ]:
        assert hasattr(mod, name)


def test_version_is_the_same_everywhere():
    # the release workflow publishes the version in pyproject.toml when a tag
    # v<version> is pushed; the package and the citation file must agree
    import re

    import nss_engine

    root = Path(__file__).resolve().parents[1]
    pyproject = re.search(
        r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(encoding="utf-8"), re.M
    )
    citation = re.search(
        r"^version: (\S+)", (root / "CITATION.cff").read_text(encoding="utf-8"), re.M
    )
    changelog = re.search(
        r"^## (\d+\.\d+\.\d+)", (root / "CHANGELOG.md").read_text(encoding="utf-8"), re.M
    )
    assert pyproject and citation and changelog
    assert pyproject.group(1) == nss_engine.__version__ == citation.group(1) == changelog.group(1)


def test_author_is_the_same_everywhere():
    # PyPI, the citation file and the website footer must name the same person;
    # to add your real name, change all three (CITATION.cff: given-names and
    # family-names, keeping the alias)
    import re

    root = Path(__file__).resolve().parents[1]
    pyproject = re.search(
        r'^authors = \[\{ name = "([^"]+)"',
        (root / "pyproject.toml").read_text(encoding="utf-8"),
        re.M,
    )
    website = re.search(
        r'^AUTHOR = "([^"]+)"', (root / "website" / "build.py").read_text(encoding="utf-8"), re.M
    )
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    given = re.search(r"given-names: \"?([^\"\n]+)", cff)
    family = re.search(r"family-names: \"?([^\"\n]+)", cff)
    alias = re.search(r"alias: \"?([^\"\n]+)", cff)
    if given and family:
        cited = f"{given.group(1).strip()} {family.group(1).strip()}"
    else:
        assert alias, "CITATION.cff needs given-names and family-names, or an alias"
        cited = alias.group(1).strip()
    assert pyproject and website
    assert pyproject.group(1) == website.group(1) == cited
