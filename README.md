# Churn_Rate_Prediction

<Project Overview>

This project aims to predict whether a user of KKBOX, a music streaming platform, will churn or not. By accurately forecasting potential cancellations, businesses can implement targeted strategies to retain customers, enhance engagement, and boost revenue.
### The criteria of "churn" is no new valid service subscription within 30 days after the current membership expires.
The project compares the performance of four popular machine learning models: Random Forest, Logistic Regression, SVM, and XGBoost. Each model is trained on both imbalanced and balanced data, and their prediction accuracy is tested on new data.

<Dataset>
The dataset used in the project is Kaggle Challenge dataset where WSDM has challenged the Kaggle ML community to help solve these problems and build a better music recommendation system.
Link to the dataset: https://www.kaggle.com/c/kkbox-music-recommendation-challenge

<GitHub Repository>
You can find the project repository at: https://github.com/SSaklecha/ChurnPred

<Requirements>
Python Version: 3.10.6
Install required packages: pip install -r requirements.txt

<Deployment Steps>
Setup:

Download and unzip the churnpred zip file. This folder contains a Code folder, a README file (this file), and the project report. The Code folder contains multiple files and subfolders (see File/Folder Descriptions for details).

1.Run pip install -r requirements.txt in your terminal to install the required packages. Ensure you are in the Code folder path (your_path/code).
2.Data Fetching:Run the getdata.py file (your_path/getdata.py) to fetch the dataset. If you prefer to use a pre-cleaned file, the final_merged file is available in the data folder. Skip to step 4 if using the pre-cleaned file.
3.EDA and Data Cleaning:Run the EDA_clean.py file to execute preprocessing steps and display EDA plots. Close each popup graph to view the subsequent plots.
4.Modeling and Evaluation:Run the main.py file (your_path/main.py).Use the iterative menu to choose which model to run next. For each model, its classification report and ROC curve are displayed.An evaluation of 1,000 unseen records in test.csv (located in the data folder) is performed, and the results are displayed.
5.Choose option 0 to exit or run a different model. Choose option 9 to see a comparison of all models. Choose 10 to run all models one after the other.

Note: All models are saved as pickle files in the model folder. When a model is selected from the menu, its pickle file is fetched and executed to show the results. Delete these pickle files if you need to rerun the models.

<File/Folder Descriptions>
data folder: Contains the data files fetched by getdata.py.
model folder: Contains the pickle files of all models.
requirements.txt: Run this file using pip to install all required packages.
EDA graphs: Contains graphs generated during EDA for reference.
main.py: Controller file. Contains the main() function.
getdata.py: Used to source the original dataset. Needs to be run manually once.
EDA_clean.py: Performs all EDA and data cleaning tasks. Needs to be run manually.
model.py: Defines a class churnmodels that includes all functions to build and evaluate the models.
feature_transformer.py: Defines a class feature_transformer that includes all functions for feature engineering, scaling, encoding, and sampling.

 
