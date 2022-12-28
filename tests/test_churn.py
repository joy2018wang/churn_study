import os
import logging
import pytest

from churn_library import import_data, perform_eda, \
	perform_feature_engineering, train_models, \
	feature_importance_plot, classification_report, encoder_helper

logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

DATA_PTH = "./data/bank_data.csv"
IMAGE_DIR = "./images/"
MODEL_DIR = "./models/"
RESPONSE = "Churn"
    
CAT_LIST = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

QUANT_LIST = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]
    
def test_import():
	'''
	test data import
	'''
	df = import_data(DATA_PTH)
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
 	df = import_data(DATA_PTH)
	try:
    	data = perform_eda(df, QUANT_LIST, CAT_LIST, IMAGE_DIR)
 	except Exception as err:
     	logging.error("Testing perfrom_eda:s")
		raise err
     
	


def test_encoder_helper():
	'''
	test encoder helper
	'''


def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''


def test_train_models():
	'''
	test train_models
	'''
