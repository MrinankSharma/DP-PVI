import itertools
import pickle
import argparse

import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.utils import as_float_array

argparser = argparse.ArgumentParser()
argparser.add_argument("--data-dir", dest="data-dir", required=True, type=str)
args = argparser.parse_args()

from ray.services import logger

def process_dataset(data_folder, filename, config, one_hot=True, should_scale=False):
    data = np.loadtxt(data_folder + "/" + filename, dtype=str, delimiter=',')

    post_string = ""

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

    mask = np.full(data.shape[1], True)
    mask[config["target"]] = False
    x = data[:, mask]

    logger.info("Reduced Data Shape {}".format(x.shape))

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, config["numerical_features"]),
        ('cat', categorical_transformer, config["categorical_features"])
    ])

    x_transformed = preprocessor.fit_transform(x)

    np.savetxt(data_folder + "/y.csv", y, delimiter=",")
    np.savetxt(data_folder + "/x" + post_string + ".csv", x_transformed, delimiter=",")

    with open(data_folder + "/preprocessor" + post_string + ".csv", "wb+") as f:
        pickle.dump(preprocessor, f)


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
        logger.info("Processing Adult Dataset with Should Scale: {} One Hot: {}".format(ss, oh))
        process_dataset(f"{args.data_dir}/adult", "adult.data", adult_config, oh, ss)
        logger.info("Processing Abalone Dataset with Should Scale: {} One Hot: {}".format(ss, oh))
        process_dataset(f"{args.data_dir}/abalone", "abalone.data", abalone_config, oh, ss)
