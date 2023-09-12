# House Price Analysis in Madrid

![Immune](https://i.imgur.com/0TSSaqL.png)

This project analyzes house prices in Madrid, Spain using Python and several machine learning libraries. The project assumes a basic understanding of data analysis and machine learning concepts, and requires the following steps to install and use:

## Installation

1. Create a Python environment using your preferred method (e.g. `conda`, `virtualenv`, etc.).
2. Activate the environment and navigate to the project directory.
3. Install the required packages using `pip` and the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

4. Install the `utils` module by running the following command from the project directory:

```bash
pip install -e src/
```

5. Start a JupyterLab server by running the following command:

```bash
jupyter lab
```

> Alternatively, you can use the `ipykernel` package to select the kernel directly from the environment inside VSCode.

## Usage

1. Navigate to the `notebooks` directory and open the desired notebook.
2. Execute the cells in the notebook to preprocess the data, perform exploratory data analysis, and build and evaluate machine learning models.
3. The data is stored in the `data` directory, which contains four subfolders:
   - `raw`: contains the raw training and testing data in CSV format.
   - `processed`: contains the processed data in CSV format.
   - `models`: contains the trained machine learning models as pickle files, along with performance metrics as JSON files.
   - `submission`: contains the submission files in CSV format.
4. The `src` directory contains a Python module with the necessary `sklearn` transformers for ETL and utility functions.
5. The `notebooks` directory contains the notebooks to execute to verify all the steps followed for the analysis of the houses in Madrid.

## Directory Structure

```bash
house_price_analysis/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── models/
│   │   ├── model_1.pkl
│   │   ├── metrics_model_1.json
│   └── submission/
│       ├── submission_1.csv
│       └── submission_2.csv
├── src/
│   ├── utils/
│   │   ├── transformers.py
│   │   ├── paths.py
│   │   ├── functions.py
│   │   └── __init__.py
│   ├── pyproject.toml
│   ├── setup.cfg
│   └── setup.py
└── notebooks/
    ├── 01_EDA.ipynb
    └── 02_Modeling.ipynb
```

This directory structure shows the organization of the project. The `data` directory contains the raw and processed data, as well as the models and submission files. The `src` directory contains the Python module with the necessary transformer and utility functions. The `notebooks` directory contains the notebooks to execute to verify all the steps followed for the analysis of the houses in Madrid.

## Data

The data used for this project is from the [Kaggle competition](https://www.kaggle.com/t/da3ba34f8e864187b05a2363a87f1cfe) "Machine Learning Avanzado I - Hands-on". The data is split into two files: `train.csv` and `test.csv`. The `train.csv` file contains the training data, which includes the target variable `buy_price_by_area`. The `test.csv` file contains the testing data, which does not include the target variable. The goal of the project is to predict the `buy_price_by_area` of the houses in the testing data.

## License

This project is licensed under the terms of the [MIT License](https://mit-license.org/).
