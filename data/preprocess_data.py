import argparse
import itertools
import logging
import pickle

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, Binarizer
from sklearn.utils import as_float_array
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Data Preprocessor")

argparser = argparse.ArgumentParser()
argparser.add_argument("--data-dir", dest="data_dir", required=True, type=str)
args = argparser.parse_args()


def process_dataset(data_folder, filename, config, one_hot=True, should_scale=False):
    try:
        data = np.loadtxt(data_folder + "/" + filename, dtype=str, delimiter=',')
        np.random.seed(0)
        data = data[np.random.permutation(data.shape[0]), :]
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
            categorical_transformer = Pipeline([('labeler', OrdinalEncoder()), ('scaler', StandardScaler())])
            post_string = post_string + "_ordinal"

        y = config["label_generator"](data[:, config["target"]].reshape(-1, 1))
        y[y == 0] = -1

        mask = np.full(data.shape[1], True)
        mask[config["target"]] = False
        x = data[:, mask]

        logger.info("Reduced Data Shape {}".format(x.shape))

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, config["numerical_features"]),
            ('cat', categorical_transformer, config["categorical_features"])
        ])

        x_transformed = preprocessor.fit_transform(x)

        logger.info(f"Final shape: {x_transformed.shape}")
        np.savetxt(data_folder + "/y.csv", y, delimiter=",")
        np.savetxt(data_folder + "/x" + post_string + ".csv", x_transformed, delimiter=",")

        with open(data_folder + "/preprocessor" + post_string + ".p", "wb+") as f:
            pickle.dump(preprocessor, f)
    except (FileNotFoundError, OSError):
        logger.warning("Missing this dataset! Not processed.")


if __name__ == "__main__":

    def adult_transformer(y):
        oe = OrdinalEncoder()
        y[y == " <=50K."] = " <=50K"
        y[y == " >50K."] = " >50K"
        y_all = oe.fit_transform(y)
        return y_all

    adult_config = {
        "numerical_features": [0, 2, 4, 10, 11, 12],
        "categorical_features": [1, 3, 5, 6, 7, 8, 9, 13],
        "folder": "/adult",
        "target": 14,
        "label_generator": adult_transformer,
        "drop_cols": [],
    }

    be = Binarizer(threshold=9.5)
    abalone_labeller = lambda z: be.fit_transform(as_float_array(z))

    abalone_config = {
        "numerical_features": [1, 2, 3, 4, 5, 6, 7],
        "categorical_features": [0],
        "folder": "/abalone",
        "target": 8,
        "label_generator": abalone_labeller,
        "drop_cols": [],
    }

    oe = OrdinalEncoder()

    bank_config = {
        "label_generator": oe.fit_transform,
        "numerical_features": [0, 11, 12, 13, 15, 16, 17, 18, 19],
        "folder": "/bank",
        "target": 20,
        "categorical_features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 14],
        "drop_cols": [10],
    }


    superconductor_labeller = lambda z: Binarizer(threshold=20.0).fit_transform(as_float_array(z))

    superconductor_config = {
        "label_generator": superconductor_labeller,
        "numerical_features": [i for i in range(81)],
        "folder": "/superconductor",
        "target": 81,
        "categorical_features": [],
        "drop_cols": [],
    }

    should_scale = [True, False]
    one_hot = [True, False]

    for ss, oh in itertools.product(should_scale, one_hot):
        logger.info("Processing Bank Dataset with Should Scale: {} One Hot: {}".format(ss, oh))
        process_dataset(f"{args.data_dir}/bank", "bank.data", bank_config, oh, ss)
        logger.info("Processing Adult Dataset with Should Scale: {} One Hot: {}".format(ss, oh))
        process_dataset(f"{args.data_dir}/adult", "adult.data", adult_config, oh, ss)
        logger.info("Processing Abalone Dataset with Should Scale: {} One Hot: {}".format(ss, oh))
        process_dataset(f"{args.data_dir}/abalone", "abalone.data", abalone_config, oh, ss)
        logger.info("Processing Superconductor Dataset with Should Scale: {} One Hot: {}".format(ss, oh))
        process_dataset(f"{args.data_dir}/superconductor", "train.csv", superconductor_config, oh, ss)
