# Install

```bash
pip install mfgqc
```

mfgQC requires **Python 3.10 or newer**. Installing it pulls in its scientific
dependencies automatically: NumPy, pandas, SciPy, Matplotlib, statsmodels, and
scikit-learn.

## Verify the install

```python
import mfgqc
print(len(mfgqc.list_analyses()), "analyses available")
```

## Development install

To work on mfgQC itself, clone the repository and install it editable with the test
extras:

```bash
git clone https://github.com/cjbrant/mfgQC
cd mfgQC
pip install -e ".[test]"
pytest            # full regression suite
```

The documentation toolchain is a separate extra:

```bash
pip install -e ".[docs]"
mkdocs serve      # live-preview the docs at http://127.0.0.1:8000
```

## Loading a CSV

mfgQC takes a pandas DataFrame, so read your file with pandas first:

```python
import pandas as pd, mfgqc
df = pd.read_csv("measurements.csv")
qc = mfgqc.load(df, measure="width")
```
