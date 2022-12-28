# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project performs data analysis of the customer churn data and run logistical model and cross-validation random forest model. It does mean value encoding for categorical attributes. After the model, it performs post-hoc diagnostic analysis such as classfication report, AUC and Shap values as well as feature importance analysis.

## Files and data description
Overview of the files and data present in the root directory. 
- README.md  README
- churn_libary.py module with functions to do the analysis
- tests/test_churn.py test file
- logs/ where log files are saved
- images ad-hoc analysis, data eda images, ...
- models/ where models are saved
- data/bank_data.csv input data set

## Running Files
### To run an example, use
``` python churn_library.py ```

Output is saved in logs/result.log
### To run the test, use
``` pytest test_churn.py ```
