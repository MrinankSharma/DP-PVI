from sacred import Ingredient
import numpy as np
import logging

dataset_ingredient = Ingredient('dataset')
logger = logging.getLogger("Dataset Generation")

@dataset_ingredient.config
def cfg():
    name = "abalone"
    scaled = True
    ordinal_cat_encoding = True
    train_proportion = 0.8
    data_base_dir = "/scratch/DP-PVI/data/"


def generate_filename(name, scaled, ordinal_cat_encoding, data_base_dir):
    ret = data_base_dir + name + "/x"
    ret_y = data_base_dir + name + "/y.csv"
    if scaled:
        ret = ret + "_scaled"

    if ordinal_cat_encoding:
        ret = ret + "_ordinal"

    ret = ret + ".csv"
    return ret, ret_y


@dataset_ingredient.capture
def load_data(name, scaled, ordinal_cat_encoding, train_proportion, data_base_dir):
    x_loc, y_loc = generate_filename(name, scaled, ordinal_cat_encoding, data_base_dir)
    logging.info(f"Using Dataset {x_loc}")
    x = np.loadtxt(x_loc, delimiter=",")
    y = np.loadtxt(y_loc, delimiter=",")
    N = x.shape[0]
    N_train = int(np.ceil(train_proportion * N))
    x_train = x[0:N_train]
    y_train = y[0:N_train]
    x_test = x[N_train:]
    y_test = y[N_train:]
    logger.info(f"Training Set: {x_train.shape[0]} examples with dimensionality {x_train.shape[1]}")
    logger.info(f"Test Set: {x_test.shape[0]} examples")

    training_set = {
        "x": x_train,
        "y": y_train,
    }

    test_set = {
        "x": x_test,
        "y": y_test
    }

    d_in = x_test.shape[1]

    return training_set, test_set, d_in
