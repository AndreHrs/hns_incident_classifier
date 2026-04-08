# Capstone project
Needs at least python 3.12 (will break if used with python < 3.12)

## Required commands
Will refine later into proper pip/conda list after production is settled
```
conda install jupytext -c conda-forge 
pip install contractions
pip install spacy
# install the lemmatization model
python -m spacy download en_core_web_sm 
```

## Code style (ruff + pre-commit)

Install the tools:
```
pip install ruff pre-commit
```

Install the pre-commit hooks (run once after cloning):
```
pre-commit install
```

After this, ruff will run automatically on every `git commit`. To run it manually:
```
# check for lint errors
python -m ruff check .

# auto-fix lint errors
python -m ruff check . --fix
```

> ⚠️ **Easy to miss:** After fixing lint errors, you must stage the changes again with `git add` before committing. **Pre-commit only checks staged files** — if the fixed files aren't staged, the commit will keep failing even though the errors are gone. *Don't make the same mistake I did* - Sincerely, Andre
