# Breast Cancer Prediction Project 🎗️

## Introduction 👋

This project focuses on building and evaluating a machine learning model to predict breast cancer using the **Wisconsin Breast Cancer Database**. The goal is to create a model that can accurately classify tumors as either **benign** or **malignant** based on a set of cellular features.

## Dataset 📊

The **Wisconsin Breast Cancer Database** was provided by Dr. William H. Wolberg from the University of Wisconsin Hospitals. The dataset was donated by Olvi Mangasarian and received by David W. Aha on July 15, 1992. The dataset contains 10 key features (attributes) and a class attribute that indicates whether an instance is benign or malignant.

### **Attribute Information**

Each instance in the dataset is described by the following attributes, scored on a scale of 1 to 10:

* Clump Thickness
* Uniformity of Cell Size
* Uniformity of Cell Shape
* Marginal Adhesion
* Single Epithelial Cell Size
* Bare Nuclei
* Bland Chromatin
* Normal Nucleoli
* Mitoses
* Class: 2 for benign, 4 for malignant

## Steps & Methodology ⚙️

The project followed a standard machine learning workflow, as detailed in the `main.ipynb` notebook:

### **1. Data Preprocessing**
* **CSV Generation**: The raw data file, `breast-cancer-wisconsin.data`, was converted into a structured CSV format (`data.csv`) with proper column names, including `id`, `clump_thickness`, `uniformity_of_cell_size`, `uniformity_of_cell_shape`, `marginal_adhesion`, `single_epithelial_cell_size`, `bare_nuclei`, `bland_chromatin`, `normal_nucleoli`, `mitoses`, and `class`.
* **Handling Missing Values**: The original dataset contained missing values denoted by a `?`. These rows were successfully removed from the dataset to create a clean CSV file, `cleaned_data.csv`.
* **Feature Engineering**: The original `class` attribute, with values `2` for benign and `4` for malignant, was converted to a binary format: `0` for benign and `1` for malignant. The `id` column was also dropped as it is not needed for the model.

### **2. Exploratory Data Analysis (EDA) & Visualization**
* **Data Overview**: The preprocessed data has a shape of (683, 10), indicating 683 instances and 10 features, with no missing values and all features converted to the correct data types.
* **Class Distribution**: The data is moderately imbalanced, with Benign cases making up 65.0% and Malignant cases at 35.0%.
* **Feature Distributions**: Kernel Density Estimate (KDE) plots were used to visualize the distribution of each feature.
* **Correlation Matrix**: A heatmap was generated to show the correlations between features.

### **3. Model Training & Evaluation**
* **Model Selection**: A variety of classification models were considered, including `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `CatBoostClassifier`, `XGBClassifier`, and `LGBMClassifier`.
* **CatBoost Model**: The `CatBoostClassifier` was chosen as the final model due to its high performance. The `catboost_best_model.pkl` file contains the best-trained model with the following parameters: `iterations=758`, `learning_rate=0.007`, `depth=3`, and `l2_leaf_reg=1.50`.
* **Evaluation Metrics**: The final model was evaluated on key metrics including **Accuracy**, **Precision**, **Recall**, and **F1-Score**, as well as a confusion matrix.

## Results 🏆

The CatBoost model showed excellent performance on the test set.

**Evaluation Metrics**:
* **Accuracy**: 96.4%
* **Precision**: 92.2%
* **Recall**: 97.9%
* **F1-Score**: 94.9%

**Confusion Matrix**:
The confusion matrix shows the model's performance in detail:
* **True Negatives (Benign)**: 85 correct predictions.
* **False Positives (Benign classified as Malignant)**: 4 incorrect predictions.
* **False Negatives (Malignant classified as Benign)**: 1 incorrect predictions.
* **True Positives (Malignant)**: 47 correct predictions.

The low number of false negatives is particularly important in medical diagnosis, as it indicates the model is highly effective at identifying malignant cases.

## Usage 🚀

To use this project, you will need to have the required libraries installed. You can then run the `main.ipynb` notebook to reproduce the results and experiment with different models.

The best-trained model is saved as `catboost_best_model.pkl`, which can be loaded and used for new predictions. You can also inspect `cleaned_data.csv` to see the preprocessed dataset used for training and evaluation.
