#  Income Prediction Workshop
This repository contains a Jupyter notebook (census_income_workshop.ipynb) that implements a machine learning model to predict whether an individual's income exceeds $50,000 based on the Census Income (Adult) dataset. The project follows a structured approach, from data preparation to model evaluation.

##  Overview
The core objective of this workshop is to build and train a neural network using PyTorch for a classification task on tabular data. The notebook performs the following key steps:

1. Data Preparation: 
Cleans the raw data, handles missing values, and separates features into categorical and continuous types. It then preprocesses the data by encoding categorical variables and standardizing continuous ones before converting them into PyTorch tensors. The dataset is split into training and testing sets.

2.  Model Design: 
Defines a custom TabularModel class using PyTorch's nn.Module. The model incorporates embeddings for categorical features and a batch-normalized layer for continuous features, followed by a hidden layer with dropout.

3. Training & Evaluation: The model is trained for 300 epochs using CrossEntropyLoss and the Adam optimizer. Its performance is then evaluated on the test set, reporting the final loss and accuracy.

4. Prediction Function (Bonus): Includes an optional function that allows a user to input new data and get a real-time prediction from the trained model.

