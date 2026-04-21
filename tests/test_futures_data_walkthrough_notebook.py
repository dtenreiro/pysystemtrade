import json
from pathlib import Path


NOTEBOOK = Path("docs/notebooks/futures_data_walkthrough.ipynb")


def load_notebook() -> dict:
    assert NOTEBOOK.exists(), f"Missing notebook: {NOTEBOOK}"
    with NOTEBOOK.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cell_sources(notebook: dict, cell_type: str) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == cell_type
    ]


def test_notebook_has_beginner_friendly_structure() -> None:
    notebook = load_notebook()
    markdown_cells = cell_sources(notebook, "markdown")
    code_cells = cell_sources(notebook, "code")

    assert notebook["nbformat"] == 4
    assert len(markdown_cells) >= 3
    assert len(code_cells) >= 3

    markdown = "\n".join(markdown_cells)
    assert "# Tutorial: Futures Data Walkthrough" in markdown
    assert "## Orientation" in markdown
    assert "## Workflow overview" in markdown
    assert "## Inspecting shipped futures data" in markdown

    first_code = code_cells[0]
    assert "def find_repo_root" in first_code
    assert '\"docs\" / \"data.md\"' in first_code
    assert "REPO_ROOT = find_repo_root()" in first_code

    runnable_code = "\n".join(code_cells)
    assert "pd.read_csv" in runnable_code
    assert "instrumentconfig.csv" in runnable_code
    assert "rollconfig.csv" in runnable_code
    assert "spreadcosts.csv" in runnable_code
    assert "DATA_ROOT / \"csvconfig\"" in runnable_code
