# Traffic-Sign-Recognition

## Data

Pickled dataset containing 32x32x3 color images split across labelled training, test and validation sets

## Environment Setup

The first step is to create an environment in side the project repository.

```
conda create -p env python==3.8 -y
conda activate env/
```

Create a .gitignore file to ignore the environment from uploding to github.

```
...
# Environments
.env
.venv
env/                         #environment name
venv/
ENV/
env.bak/
venv.bak/
...
```

## Project Setup

```
Traffic-Sign-Recognition
│   .gitignore
│   README.md
│   requirements.txt
│   setup.py
│
└───src
│   │   __init__.py
│   │   exception.py
│   │   logger.py
│   │   utils.py
│   │
│   └───components
│   │   │   __init__.py
│   │   │   data_ingestion.py
│   │   │   data_transformation.py
│   │   │   model_trainer.py
│   │
│   └───pipeline
│       │   __init__.py
│       │   predict_pipeline.py
│       │   train_pipeline.py
│

```
