# Avarto_Customer_Segmentation
Customer Segmentation for Avarto Finances


1. Installation
2. Summary
3. File Descriptions
4. Results
5. Licensing, Authors, and Acknowledgements
 
# Installation:
The libraries used in the project are as follows:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Sklearn

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.


# Summary:
The goals of the project are divided into three parts:

- Customer Segmentation: Comparing the demographic data of customers of a mail-order company with general population demographics data to identify the portion of the population which could be potential future customers. 
This task is performed unsupervised leanring techniques like kmeans and PCA. 

- Supervised Machine Learning Model: Building a supervised machine learning model using the demographics training dataset of the customers. 
Regression boosting algorithms like Gradient Boosting, Ada Boosting and XGBoosting are used to train the model. Purpose of the model to classify new data points into potential targets or non_targets

- Kaggle Competition: On evaluating the supervised learning model with ROC_AUC, the predictions for the demographics test data have to be submitted to Kaggle. 
Currently an accuracy of 0.79959 is achieved using fine-tuned Gradient Boosting Algorithm. 

![picture](https://github.com/RuchitDoshi/Avarto_Customer_Segmentation/blob/master/Images/kaggle.png)


# File Descriptions:
Since, the data is private, it cannot be shared here. The data is stored in a folder named 'capstone_data' in the same directory as this project. There 5 data files in the project :
- Udacity_AZDIAS_052018.csv: demographics data of general population in a city of Germany
- Udacity_CUSTOMERS_052018.csv: demographics data of customers of a mail order company
- Udacity_MAILOUT_052018_TRAINING.csv: demographics data for individual who were targets_training
- Udacity_MAILOUT_052018_TEST.csv: demographics data for individual who were targets_testing
- DIAS Attributes - Values 2017.xlsx: Descritpion of features and their values

Due to large space requirements of the dataset, the work of this project was divided across two files. However, these files are independent of each other.
The description of the workbook files is as follows:

- Customer_Segmentation.ipynb: This workbook focusses on Preprocessing of azdias and customer datasets and implementation of Unsupervised Learning models. 
It answers the Customer Segment part of this project.

- Supervised_learning.ipynb : This workbook focussses on Preprocessing of training and testing datasets and implementation of supervised learning models.
The hyper-parameter tuning and generates the submission file needed for kaggle.

# Results:
The main findings of this project can be found [here](https://medium.com/@ruchitsd/customer-segmentation-and-supervised-learning-model-for-avarto-financial-solutions-374c946c99ac).

# Licensing, Authors, and Acknowledgements:
Must give credit to Avarto Financial Solutions for the data. The data is private and cannot be shared with anyone. The workbooks here are only for exploration and cannot used for any other reasons.














