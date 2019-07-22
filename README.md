# DP-PVI

Note: we use `torch=1.1.0` for compatability with Pipenv.
  OSX users may need to install `libomp` e.g. `brew install libomp`
Note: for slack notifications, you will need to add a slack webhook into a file name `slack_webhook` in the base project directory.

## Datasets
`preprocess_data.py` expects the UCI datasets in the `data/abalone` and the `data/adult` directories to be pre-processing. Each dataset then creates a labels file, `y.csv` and additionally several data files, `x_ordinal.csv, x_scaled.csv` and `x_scaled_ordinal.csv`

  