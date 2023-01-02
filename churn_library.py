'''
Module to perform churn analysis and train models
Author: Joy Wang
File created: Dec 12, 2022
Last updated: Dec 28, 2022
'''


from typing import Optional, List
import logging

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(path_to_csv: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    Parameters:
    -----------
            path_to_csv: a path to the csv
    Returns:
    --------
            df: pandas dataframe
    '''
    df = pd.read_csv(path_to_csv)
    return df


def perform_eda(
        df: pd.DataFrame,
        quant_columns: List[str],
        cat_columns: List[str],
        image_folder: str):
    '''
    perform eda on df and save figures to images folder
    Parameters:
    -----------
            df: pandas dataframe
            quant_columns: list of continuous columns
            cat_columns: list of categorical colums
            image_folder: folder to save image
    '''
    logging.info(df.head())
    logging.info(df.shape)
    logging.info(df.isnull().sum())
    logging.info(df.describe())

    # save hist of churn
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(f'{image_folder}churn_hist.png')
    plt.close()

    # save normalized bar chart for categorical columns
    for col in cat_columns:
        df[col].value_counts('normalize').plot(kind='bar')
        plt.savefig(f'{image_folder}{col}_bar.png')
        plt.close()

    # save hist of continuous columns
    for col in quant_columns:
        df[col].hist()
        plt.savefig(f'{image_folder}{col}_hist.png')
        plt.close()

    # save heatmap for correlation
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f'{image_folder}corr_heatmap.png')
    plt.close()


def encoder_helper(
        df: pd.DataFrame,
        cat_columns: List[str],
        response: Optional[str] = "y") -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    Parameters:
    -----------
            df: pandas dataframe
            cat_columns: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                for naming variables or index y column]

    Returns:
    --------
            df: pandas dataframe with new columns for
    '''
    for col in cat_columns:
        df[f"{col}_Churn"] = df.groupby(col)[response].transform('mean')

    return df


def perform_feature_engineering(
        df: pd.DataFrame,
        x_cols: List[str],
        response: Optional[str] = "y"):
    '''
    feature engineering

    Parameters:
    -----------
              df: pandas dataframe
              x_cols: the X columns for the model
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    Returns:
    --------
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    X = df[x_cols]
    y = df[response]

    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.DataFrame, y_test: pd.DataFrame,
                                model_folder: str, image_folder: str):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    Parameters:
    -----------
            X_train: training  attribute values
            X_test:  test attribute values
            y_train: training response values
            y_test:  test response values
            model_folder: folder which stores the models
            image_folder: folder which stores the images
    '''
    cv_rfc_model = joblib.load(f'{model_folder}cv_rfc_model.pkl')
    lr_model = joblib.load(f'{model_folder}logistic_model.pkl')

    model_dict = {
        'Logistic Regression': lr_model,
        'Random Forest': cv_rfc_model}

    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.close()

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(f"{image_folder}rfc_roc.png")
    plt.close()

    plt.figure(figsize=(15, 8))
    explainer = shap.TreeExplainer(cv_rfc_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig(f"{image_folder}rfc_shap.png")
    plt.close()

    # score
    for model_type, model in model_dict.items():
        y_train_preds = model.predict(X_train)
        y_test_preds = model.predict(X_test)
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1, str(f'{model_type} Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.1, str(classification_report(y_test, y_test_preds)),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.5, str(f'{model_type} Test'), {'fontsize': 10},
                 fontproperties='monospace')
        plt.text(0.01, 0.6, str(classification_report(y_train, y_train_preds)),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')

        plt.savefig(f"{image_folder}{model_type}classification_report.png")
        plt.close()


def feature_importance_plot(
        model,
        col_names: List[str],
        image_folder: str,
        model_name: str):
    '''
    creates and stores the feature importances in pth
    Parameters:
    -----------
            model: model object containing feature_importances_
            col_names: name list of features
            image_folder: path to store the figure
            model_name: the model_name in the saved file
    '''
    # Calculate feature importances
    feature_importance = model.feature_importances_
    num_features = len(feature_importance)
    # Sort feature importances in descending order
    indices = np.argsort(feature_importance)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    x_names = [col_names[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(num_features), feature_importance[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(num_features), x_names, rotation=90)

    plt.savefig(f"{image_folder}feature_importance_{model_name}.png")
    plt.close()


def train_models(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        model_folder: str):
    '''
    train, store model results: store models
    Parameters:
    -----------
              X_train: X training data
              y_train: y training data
              model_folder: the folder where models are saved
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    joblib.dump(cv_rfc.best_estimator_, f'{model_folder}cv_rfc_model.pkl')
    joblib.dump(lrc, f'{model_folder}logistic_model.pkl')


if __name__ == "__main__":

    logging.basicConfig(
        filename='./logs/results.log',
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')

    DATA_PTH = "./data/bank_data.csv"
    IMAGE_EDA_DIR = "./images/eda/"
    IMAGE_RESULTS_DIR = "./images/results/"
    MODEL_DIR = "./models/"
    RESPONSE = "Churn"

    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_lst = [
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

    KEEP_COLS = [
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
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    data = import_data(DATA_PTH)
    perform_eda(data, quant_lst, category_lst, IMAGE_EDA_DIR)

    data = encoder_helper(data, category_lst, RESPONSE)

    X_train_df, X_test_df, y_train_df, y_test_df = perform_feature_engineering(
        data, KEEP_COLS, RESPONSE)

    train_models(X_train=X_train_df, y_train=y_train_df,
                 model_folder=MODEL_DIR)

    classification_report_image(X_train=X_train_df, X_test=X_test_df,
                                y_train=y_train_df, y_test=y_test_df,
                                model_folder=MODEL_DIR, image_folder=IMAGE_RESULTS_DIR)

    feature_importance_plot(joblib.load(f'{MODEL_DIR}cv_rfc_model.pkl'),
                            X_train_df.columns, IMAGE_RESULTS_DIR, 'rfc')
