import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "tour.ipynb"


def test_tour_notebook_is_valid_python():
    nb = json.loads(NOTEBOOK.read_text())
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
