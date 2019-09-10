# Differentially Private Federated Variational Inference
**Mrinank Sharma, Michael Hutchinson, Siddharth Swaroop, Antti Honkela, Richard Turner**

This file contains instructions for reproducing the  plots found in our NeurIPS 2019 Privacy in Machine Learning (PiML) workshop.

1. Download the UCI Adult Dataset, which can be found [here](https://archive.ics.uci.edu/ml/datasets/Adult). You will need to manually merge the training and test data in the data folder, found in files `adult.data` and `adult.test` respectively. Merge these into one file called `adult.data` and place this in the a folder called `data` which should have the following file structure.

    ```
    _path_to_folder
        - data
            - adult
                - adult.data
    ```

2. Ensure that you have all of the dependencies installed, which can be managed with `pipenv`. Additionally, you will also need to set up a MongoDB instance. The code assumes that this will be located at `localhost:9001` (you could use ssh port-forwarding to accomplish this). If this is not the case, you will need to modify the `MongoDBOption.py` file appropriately. 

3. Run the Python file `data/preprocess_data.py` with the `data_dir ` argument i.e. `python preprocess_data.py --data-dir _path_to_folder/data`.

4. Modify the file `experiments/ingredients/dataset_ingredient.py`, modifying line 26 to read `data_base_dir="_path_to_folder/data`. Additionally, if you desire, modify files `batch_vi_experiment.py, datapoint_experiment.py, dp_batch_vi_experiment.py, PVI_experiment.py`, all of which are in the `experiments/workshop` directory to have `logging_base_directory` to point to a location of your choice.

5. Use the file `experiments/dispatch/simple_dispatcher.py` to run the experiment config files found in the `experiments/workshop/experiment_configs` folder. This can be done as follows:
    ```
    python experiments/dispatch/simple_dispatcher.py --num-cpus 2 --exp-file experiments/workshop/experiment_configs/DP-PVI_Investigation/*
    python experiments/dispatch/simple_dispatcher.py --num-cpus 2 --exp-file experiments/workshop/experiment_configs/Inhomogeneity_Investigation/*
    ```
    Note that the `num-cpus` options determines the number of individual runs which are run at once.
 
6. Run the notebooks found in `workshop/notebooks`. 

If these instructions are not clear in any way, or you are unable to reproduce these results, please contact Mrinank (`mrinank.sharma@eng.ox.ac.uk`) or Michael (`michael.hutchinson@univ.ox.ac.uk`).

