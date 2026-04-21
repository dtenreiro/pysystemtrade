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


def execute_code_cells(code_cells: list[str]) -> None:
    namespace: dict[str, object] = {}
    for index, code in enumerate(code_cells):
        exec(compile(code, f"<futures-data-walkthrough-cell-{index}>", "exec"), namespace)


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


def test_notebook_references_repo_files_and_interfaces() -> None:
    notebook = load_notebook()
    code_cells = cell_sources(notebook, "code")
    all_text = "\n".join(cell_sources(notebook, "markdown")) + "\n" + "\n".join(
        code_cells
    )

    assert "data/futures/csvconfig/instrumentconfig.csv" in all_text
    assert "data/futures/csvconfig/rollconfig.csv" in all_text
    assert "data/futures/roll_calendars_csv/BUND.csv" in all_text
    assert "data/futures/multiple_prices_csv/BUND.csv" in all_text
    assert "data/futures/adjusted_prices_csv/BUND.csv" in all_text
    assert "## Mapping document concepts to Python objects" in all_text
    assert "## Interfaces and entry points" in all_text
    assert "csvFuturesContractPriceData" in all_text
    assert "csvRollCalendarData" in all_text
    assert "csvFuturesMultiplePricesData" in all_text
    assert "csvFuturesAdjustedPricesData" in all_text
    assert "csvFuturesSimData" in all_text
    assert "dbFuturesSimData" in all_text
    assert "dataBlob" in all_text
    assert "from sysdata.data_blob import dataBlob" not in all_text
    assert "from sysdata.sim.csv_futures_sim_data import csvFuturesSimData" not in all_text
    assert "from sysdata.sim.db_futures_sim_data import dbFuturesSimData" not in all_text

    execute_code_cells(code_cells)

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
