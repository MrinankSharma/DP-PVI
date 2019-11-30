# Differentially Private Paritioned Variational Inference

## Paper
Please find [the paper here](https://arxiv.org/pdf/1911.10563.pdf). 

## Reproducibility
Requirements:

* Python 3.7
* We use [Pipenv](https://pipenv.readthedocs.io/en/latest/) to handle Python package installations. Note: we use `torch=1.1.0` for compatability with Pipenv. The `Pipfile.lock` can be used to ensure that you are running with the exact same dependencies as we have used. 
* OSX users may need to install `libomp` e.g. with Homebrew `brew install libomp`
* A [MongoDB](https://www.mongodb.com/) is used to store results, and Sacred ([Github](https://github.com/IDSIA/sacred)) is used to place useful information in this database. 

Please see the `experiments/workshop` directory for a detailed description of how to reproduce plots found in our NeurIPS 2019 Privacy in Machine Learning (PiML) Workshop Submission. 

## Datasets
`preprocess_data.py` expects the UCI datasets ([here](https://archive.ics.uci.edu/ml/datasets.php)) in the `data/abalone` and the `data/adult` directories to be pre-processing. Each dataset then creates a labels file, `y.csv` and additionally several data files, `x_ordinal.csv, x_scaled.csv` and `x_scaled_ordinal.csv`

## Logging 
The Sacred ([Github](https://github.com/IDSIA/sacred)) logging functionality relies on the existence of a Mongo database, by default assumed to be in `localhost:9001`. If this is on another machine, ssh port forwarding is an easy way to get this working without changing the code. (E.g. for mlg people, if the database is on Hinton: `ssh hinton -L 9001:localhost:9001` will connect port 9001 on Hinton to port 9001 on your machine. In order to bring up and bring down the database, suitable scripts are in `/scratch/mjh252/mongo`)

Omniboard ([Github](https://github.com/vivekratnavel/omniboard)) is an excellent way to view results in a Sacred database.

Deleting database entries can be done as follows, in the mongo shell:
```angular2
use sacred # use the correct database name
db.runs.remove({'config.experiment_tag':'....'}) # apply the correct filter
db.metrics.remove({run_id: {$gt: 333, $lt: 600}}) # remove metrics, linked by run id
```


**Beware**: whilst this removes runs from the omniboard view window, it will not actually delete the log files!!
  
