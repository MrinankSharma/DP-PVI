import itertools

import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.utils import as_float_array

def process_dataset(data_folder, filename, config, one_hot=True, should_scale=False):
    data = np.loadtxt(data_folder + "/" + filename, dtype=str, delimiter=',')

    post_string = "/x"

    # grab the numerical part of the array and convert to a float
    if should_scale:
        numerical_transformer = StandardScaler()
        post_string = post_string + "_scaled"
    else:
        numerical_transformer = StandardScaler(with_mean=False, with_std=False)

    if one_hot:
        categorical_transformer = OneHotEncoder(sparse=False)
    else:
        categorical_transformer = OrdinalEncoder()
        post_string = post_string + "_ordinal"

    y = config["label_generator"](data[:, config["target"]].reshape(-1, 1))
    x = data[:, 0:14]

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, config["numerical_features"]),
        ('cat', categorical_transformer, config["categorical_features"])
    ])

    x_transformed = preprocessor.fit_transform(x)
    x_transformed.shape

    np.savetxt(data_folder + "/y.csv", y)
    np.savetxt(data_folder + post_string + ".csv", x_transformed)


if __name__ == "__main__":

    oe = OrdinalEncoder()
    adult_config = {
        "numerical_features": [0, 2, 4, 10, 11, 12],
        "categorical_features": [1, 3, 5, 6, 7, 8, 9, 13],
        "folder": "/abalone",
        "target": 14,
        "label_generator": oe.fit_transform
    }

    be = Binarizer(threshold=10.0)
    abalone_labeller = lambda z: be.fit_transform(as_float_array(z))

    abalone_config = {
        "numerical_features": [1, 2, 3, 4, 5, 6, 7],
        "categorical_features": [0],
        "folder": "/abalone",
        "target": 8,
        "label_generator": abalone_labeller
    }

    should_scale = [True, False]
    one_hot = [True, False]

    for ss, oh in itertools.product(should_scale, one_hot):
        process_dataset("/Users/msharma/workspace/DP-PVI/data/adult", "adult.data", adult_config, oh, ss)
        process_dataset("/Users/msharma/workspace/DP-PVI/data/abalone", "abalone.data", abalone_config, oh, ss)
