import json
from pathlib import Path


NOTEBOOK = Path("docs/notebooks/futures_data_walkthrough.ipynb")


def load_notebook() -> dict:
    assert NOTEBOOK.exists(), f"Missing notebook: {NOTEBOOK}"
    with NOTEBOOK.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def joined_sources(notebook: dict, cell_type: str) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == cell_type
    )


def test_notebook_has_beginner_friendly_structure() -> None:
    notebook = load_notebook()
    markdown = joined_sources(notebook, "markdown")

    assert "# Tutorial: Futures Data Walkthrough" in markdown
    assert "new `pysystemtrade` users" in markdown
    assert "## Orientation" in markdown
    assert "## Workflow overview" in markdown
    assert "## Inspecting shipped futures data" in markdown
