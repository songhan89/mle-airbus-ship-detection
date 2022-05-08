airbus-ship-detection
==============================

ML Engineering project for airbus ship detection.

Visit this Kaggle page for more info
* https://www.kaggle.com/c/airbus-ship-detection

## Getting Started

### Airbus Project

There are two ways to experiment with the dataset:

* Sign up Jupyter notebook account on Kaggle, add data "_Competition Data -> Air Bus Ship Detection Challenge_"
* Download the 30gb dataset using Kaggle API https://github.com/Kaggle/kaggle-api

### GCP guide

* https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai

## Sample notebook

* https://www.kaggle.com/code/kmader/baseline-u-net-model-part-1
* https://www.kaggle.com/code/kmader/from-trained-u-net-to-submission-part-2/notebook

## Alternative Dataset

* We can use trained model from the Airbus Kaggle challenge to do transfer learning on Sentinel-2 satellite (10-m resolution)
* Sample data around Singapore can be found here on [EO Hub Browser](https://apps.sentinel-hub.com/eo-browser/?zoom=13&lat=1.24101&lng=103.82303&themeId=DEFAULT-THEME&visualizationUrl=https%3A%2F%2Fservices.sentinel-hub.com%2Fogc%2Fwms%2F42924c6c-257a-4d04-9b8e-36387513a99c&datasetId=S2L1C&fromTime=2020-05-25T00%3A00%3A00.000Z&toTime=2020-05-25T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR) 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
