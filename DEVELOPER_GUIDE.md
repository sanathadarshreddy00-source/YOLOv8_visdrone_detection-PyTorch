# Developer Guide (short)

Purpose: quick developer notes to orient contributors and reduce friction when running scripts locally.

Important locations
- `configs/experiment_config.yaml`: training/eval hyperparameters.
- `configs/paths.yaml`: canonical paths (edit carefully).
- `src/`: reusable modules (utilities, data loaders, helpers).

Running locally
- Prefer creating a reproducible environment (conda or venv). Example:

```powershell
conda create -n cv310 python=3.10
conda activate cv310
pip install -r requirements.txt   # if present
$env:PYTHONPATH = "$(Get-Location)"
```

Best practices
- Do not change `configs/paths.yaml` without coordinating; use backups and manifest for destructive moves.
- Use `paths` from `src.utils` rather than hard-coded paths.
- Write small unit tests for new utilities and add them to a `tests/` folder.

Suggested next improvements (low effort, high impact)
- Add `requirements.txt` or `environment.yml` and commit it.
- Add `pyproject.toml` or `setup.cfg` and make the project `pip install -e .` to avoid `PYTHONPATH` hacks.
- Add a small GitHub Actions CI: lint + run `tools/generate_directory_record.py` + basic imports.
