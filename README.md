# Lab 6 Repo Check

This repository now contains:

- the original Adult/Census TFDV lab
- a custom Telco Churn version suitable for a Lab 6 submission

## What is in the repo

- `TFDV_Lab1.ipynb`: the full lab notebook
- `TFDV_Lab6_Telco_Churn.ipynb`: custom notebook using Telco Churn data
- `util.py`: helper that injects malformed rows into the evaluation split
- `data/adult.data`: source dataset used by the notebook
- `data/telco_churn.csv`: Telco Churn dataset used by the custom notebook
- `slice_sample.csv`: generated CSV used for slice-based TFDV stats
- `img/tfdv.png`: image asset

## Current notebook flow

The notebook currently does the following:

1. Loads `data/adult.data` into a pandas DataFrame.
2. Splits the data into train and evaluation sets.
3. Injects a few bad rows into the evaluation set with `add_extra_rows()`.
4. Generates TFDV statistics for the training data.
5. Infers a schema from the training statistics.
6. Compares evaluation statistics against training statistics.
7. Cleans some evaluation issues and validates anomalies against the schema.
8. Revises the schema to tolerate or explicitly accept selected values.
9. Creates slices for subgroup analysis and compares them visually.

## Repo risks and observations

- The notebook is tightly coupled to the Adult dataset schema.
- `util.py` previously used `DataFrame.append()`, which is removed in pandas 2.x.
- There is no environment file such as `requirements.txt`, `pyproject.toml`, or `environment.yml`.
- The notebook metadata targets Python 3.10, but your machine currently exposes Python 3.12 launchers, so dependency/version drift is likely.
- `slice_sample.csv` is generated output, not the real source-of-truth dataset.

## Hardcoded Adult-dataset assumptions

If you want to submit this lab using your own dataset, these are the places that must change:

- Notebook cell 5: reads `data/adult.data`
- Notebook cell 10 and `util.py`: injects Adult-specific bad rows with fields like `age`, `occupation`, `native-country`, and `label`
- Notebook cell 25: filters invalid `age` values specifically
- Notebook cell 33: relaxes schema rules for `native-country` and `occupation`
- Notebook cell 35: manually adds `Asian` to the `race` domain
- Notebook cell 37: restricts the `age` range to `17..90`
- Notebook cells 42, 53: slices on `sex` and `race`

## Best way to customize this lab

Pick a tabular CSV dataset with:

- a mix of numeric and categorical columns
- a clear target column
- a few columns suitable for slicing, such as `gender`, `class`, `region`, `plan`, or `status`
- enough rows for train/eval splits to be meaningful

Good dataset styles for this lab:

- Titanic survival
- Bank marketing
- Telco churn
- Loan approval
- Student performance

## Recommended customization pattern

Use the same notebook structure, but swap the dataset-specific logic:

1. Replace cell 5 with a loader for your CSV and verify column names.
2. Rewrite `add_extra_rows()` in `util.py` so the injected bad rows match your new dataset columns.
3. Replace the cleaning logic in cell 25 with rules for one or two numeric columns in your dataset.
4. Replace the schema edits in cells 33, 35, and 37 with domains/ranges that make sense for your features.
5. Replace the slicing logic in cells 42 to 59 with subgroup comparisons that fit your dataset.

## Example mapping if you use Titanic

Possible feature substitutions:

- `age` -> keep `Age`
- `sex` -> keep `Sex`
- `race` -> replace with `Pclass`
- `native-country` -> replace with `Embarked`
- `occupation` -> replace with `Cabin` presence or another categorical field
- `label` -> replace with `Survived`

Possible anomaly injections:

- `Age = -5`
- `Fare = 99999`
- `Sex = "Unknown"`
- `Embarked = "X"`
- missing values in one important column

Possible slices:

- `Sex`
- `Pclass`
- `Sex + Pclass`

## Suggested submission approach

For a clean "own mix" submission:

1. Rename the notebook to something specific to your dataset.
2. Add 2 to 4 short markdown cells explaining why you chose the dataset.
3. Keep the same TFDV concepts: stats, schema inference, anomaly detection, schema updates, and slicing.
4. Make sure your injected anomalies and schema revisions are clearly tied to your dataset, not copied from Adult.

## What I changed

- Updated `util.py` to use a pandas-2-compatible row append helper.

## Next practical step

The Telco version is now added in `TFDV_Lab6_Telco_Churn.ipynb`. If you want, I can still do one more pass and tighten the notebook narrative for a polished submission.
