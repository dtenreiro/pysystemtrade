# Futures Data Walkthrough Notebook Design

Date: 2026-04-21
Status: Draft approved for spec review
Target artifact: `docs/notebooks/futures_data_walkthrough.ipynb`
Source document: `docs/data.md`

## Goal

Create a tutorial-style Jupyter notebook that showcases the contents of `docs/data.md` for new `pysystemtrade` users. The notebook should explain the futures-data pipeline in plain language while also using runnable code cells to inspect the repository's actual data files, configuration assets, and Python objects.

## Audience

The primary audience is new `pysystemtrade` users who need a guided tour of the futures-data workflow, the shipped data assets, and the code paths that implement the concepts described in `docs/data.md`.

## Success Criteria

The notebook is successful if it:

1. Gives a beginner-friendly walkthrough of the four major parts of `docs/data.md`.
2. Runs through its main path in a fresh checkout using shipped repository content only.
3. Uses code cells to inspect real futures-data files and importable objects from this repo.
4. Clearly distinguishes required runnable content from optional external integrations such as MongoDB and Interactive Brokers.
5. Fits the documentation tone and placement of the existing notebook under `docs/notebooks/`.

## Constraints

- The main tutorial path must not depend on local databases, broker connections, or credentials.
- The notebook should favor lightweight inspection over expensive data generation or end-to-end system runs.
- Optional integration examples must be explicitly guarded so a fresh checkout still executes cleanly.
- The notebook should teach from the actual repository layout rather than synthetic examples.

## Recommended Approach

Build a hybrid tutorial notebook. The main sections stay fully runnable using shipped docs, CSV-backed futures data, and importable Python modules. A final optional section shows the shape of external interfaces behind guarded cells so the notebook still reflects the complete scope of `docs/data.md`.

This approach is preferred over a fully static documentation notebook because the user explicitly wants runnable inspection cells. It is preferred over a contributor-oriented notebook because the target reader is a beginner, not a maintainer.

## Notebook Structure

The notebook will be created at `docs/notebooks/futures_data_walkthrough.ipynb`.

It will follow this sequence:

### 1. Orientation

- Introduce the notebook's purpose and audience.
- Explain that it is a guided companion to `docs/data.md`.
- Include a small code cell that reads `docs/data.md` and extracts the major headings so readers can see the original source outline.

### 2. Workflow Overview

- Summarize Part 1 of `docs/data.md`: the path from static configuration and historical contract prices to roll calendars, multiple prices, adjusted prices, and final sim/production inputs.
- Include cells that point to the repo locations involved in that workflow, such as:
  - `data/futures/`
  - `data/futures/csvconfig/`
  - roll calendar CSV directories
  - multiple price CSV directories
  - adjusted price CSV directories
- Keep this section descriptive and navigational rather than computationally heavy.

### 3. Inspecting Shipped Futures Data

- Use small `pathlib` and `pandas` examples to inspect representative files from the shipped dataset.
- Show examples of:
  - instrument configuration or spread-cost CSVs
  - a sample roll calendar
  - a sample multiple prices file
  - a sample adjusted prices file
- Focus on columns, index structure, and how each file type fits into the documented workflow.

### 4. Mapping Concepts to Python Objects

- Import a small, stable set of modules or classes referenced by `docs/data.md`.
- Show where those objects live in the repo and how they correspond to concepts in the document.
- Prefer lightweight introspection such as class names, docstrings, module paths, or selected method names over deep execution.
- The goal is to bridge the prose document to the codebase without requiring readers to know the internals first.

### 5. Interfaces and Entry Points

- Explain the transition from stored data objects to sim and production interfaces, as described in Part 4 of `docs/data.md`.
- Use importable objects and module inspection to show where interface layers are defined.
- Keep examples concrete enough that a new user can tell where to look next in the codebase.

### 6. Optional External Integrations

- Add clearly labeled optional cells for MongoDB and Interactive Brokers.
- Guard those cells with `try/except` or environment checks so failure to import or connect does not break the notebook.
- Use these cells to demonstrate interface shape only, not to require live credentials or services.
- Make it explicit in markdown that these examples are optional and may not run in a default environment.

## Content and Teaching Style

- Markdown cells should be concise and oriented toward beginners.
- Each code cell should demonstrate one idea only.
- The notebook should explain why each file or object matters in the overall futures-data pipeline.
- Large outputs should be trimmed to a few rows or a short summary.
- The notebook should avoid becoming a duplicate of `docs/data.md`; it should be a guided runnable companion.

## Data Flow in the Notebook

The notebook's teaching flow should mirror the data flow described in the source document:

1. Static configuration and source data locations
2. Individual futures contract data
3. Roll calendars
4. Multiple prices
5. Adjusted prices
6. Spot FX and higher-level simulation or production interfaces

Each step should explicitly connect the previous data layer to the next so readers understand dependency order, not just file names.

## Error Handling

- Required cells should use repository-local paths and imports that are expected to work in a fresh checkout.
- Cells that depend on optional packages, services, or environment setup must fail gracefully with explanatory text.
- If a file path or import varies by environment, the notebook should state what assumption it is making.
- Optional sections should not prevent kernel execution for the core tutorial path.

## Verification Plan

The implementation should verify the notebook at two levels:

1. Structural validation
   - Confirm the notebook JSON is valid after generation and edits.
   - Confirm the file is written to `docs/notebooks/futures_data_walkthrough.ipynb`.

2. Execution validation
   - Run the main notebook path top-to-bottom in the local repo environment if dependencies allow.
   - If full notebook execution is not possible, run representative import and file-inspection cells separately and report the limitation clearly.
   - Confirm optional MongoDB and IB cells are guarded rather than assumed to work.

## Out of Scope

- Building a full backtest or production trading workflow inside the notebook
- Downloading external data
- Requiring MongoDB or Interactive Brokers setup for the main path
- Rewriting `docs/data.md`
- Teaching every internal implementation detail of the futures-data subsystem

## Implementation Notes

- Use the `jupyter-notebook` scaffold flow so the notebook starts from the tutorial template rather than handwritten JSON.
- Prefer stable filenames and repository-relative references.
- Keep the notebook under version control as documentation, not under `output/`.

## Open Questions

No unresolved questions remain from the approved design. The notebook should proceed with the hybrid tutorial approach described above.
