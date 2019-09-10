In order to preprocess the standard UCI datasets considered, please run the `preprocess_data.py` file found in the `/data` folder with `--data-dir path_to_data_folder`. The data folder should have the following structure:
```
data
    - abalone
        - abalone.data
    - adult
        - adult.data
    - bank
        - bank.data
    - superconductivity
        - superconductivity.data
```
The python file will then produce `x.csv` and `y.csv` files, considering combinations of categorical features being one-hot encoded and features being scaled to have zero mean and unit variance.