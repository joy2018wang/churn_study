'''
functions to do tests
Author: Joy Wang
File created: Dec 12, 2022
Last updated: Dec 31, 2022
'''
import sys
import os
import logging
import pandas as pd

sys.path.append(".")

from churn_library import import_data, perform_eda, \
    perform_feature_engineering, train_models, encoder_helper

logging.basicConfig(
    filename='./logs/test_churn_study.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

DATA_PTH = "./data/bank_data.csv"
IMAGE_DIR = "./images/"
MODEL_DIR = "./models/"
RESPONSE = "Churn"


def test_import():
    '''
    test data import
    '''
    ori_data = import_data(DATA_PTH)
    try:
        assert ori_data.shape[0] > 0
        assert ori_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    ori_data = import_data(DATA_PTH)
    try:
        perform_eda(ori_data, ['Customer_Age', 'Dependent_count'],
                           ['Gender', 'Education_Level'], IMAGE_DIR)
    except Exception as err:
        logging.error("Testing perfrom_eda:there is an error")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    fake_data = pd.DataFrame({'x1': [1, 1, 2, 2, 2], 'x2': ['a', 'b', 'a', 'b', 'a'],
                       'y': [0, 1, 0, 0, 1]})
    fake_data = encoder_helper(fake_data, ['x1', 'x2'], "y")
    assert (fake_data.columns == ['x1', 'x2', 'y', 'x1_Churn', 'x2_Churn']).all()
    assert (fake_data['x1_Churn'] - [.5, .5, .33, .33, .33]).all() < 0.01
    assert (fake_data['x2_Churn'] - [.33, .5, .33, .5, .33]).all() < 0.01


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    fake_data = pd.DataFrame({'x1': [1, 1, 2, 2, 2], 'x2': [
                      'a', 'b', 'a', 'b', 'a'], 'y': [0, 1, 0, 0, 1]})
    ind_train, ind_test, res_train, res_test = \
        perform_feature_engineering(fake_data, ['x1'], 'y')
    assert (ind_train.columns == ['x1']).all()
    assert ind_train.shape[0] == 3
    assert ind_test.shape[0] == 2
    assert res_train.shape[0] == 3
    assert res_test.shape[0] == 2


def test_train_models():
    '''
    test train_models
    '''
    category_lst = [
        'Gender',
        'Education_Level',
    ]

    quant_lst = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
    ]

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Gender_Churn',
        'Education_Level_Churn']

    ori_data = import_data(DATA_PTH)
	# remove previous model files
    for file_name in ['cv_rfc_model.pkl', 'logistic_model.pkl']:
        file_name_with_path = f'{MODEL_DIR}{file_name}'
        if os.path.exists(file_name_with_path):
            os.remove(file_name_with_path)

    perform_eda(ori_data, quant_lst, category_lst, IMAGE_DIR)

    ori_data = encoder_helper(ori_data, category_lst, RESPONSE)

    ind_train_df, _, res_train_df, _ = perform_feature_engineering(
        ori_data, keep_cols, RESPONSE)

    train_models(X_train=ind_train_df, y_train=res_train_df,
                 model_folder=MODEL_DIR)

    # verify models are saved
    for file_name in ['cv_rfc_model.pkl', 'logistic_model.pkl']:
        assert os.path.exists(f'{MODEL_DIR}{file_name}')
