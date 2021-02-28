"""
                Analytics Vidhya Jobathon

File Description: Utils + Constants
Date: 27/02/2021                
Author: vishwanath.prudhivi@gmail.com
"""

#import required libraries
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler 
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from sklearn import manifold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#global constants
PRIMARY_KEY = 'ID'
TARGET = 'Response'
#FEATURES = None
RAW_TRAIN_DATA_PATH = r'C:\Users\vishw\Desktop\Job Prep\Analytics Vidhya - Health Insurance\train_Df64byy.csv'
RAW_TEST_DATA_PATH = r'C:\Users\vishw\Desktop\Job Prep\Analytics Vidhya - Health Insurance\test_YCcRUnU.csv'
PROCESSED_TRAIN_DATA_PATH = r'C:\Users\vishw\Desktop\Job Prep\Analytics Vidhya - Health Insurance\train_processed.csv'
PROCESSED_TEST_DATA_PATH = r'C:\Users\vishw\Desktop\Job Prep\Analytics Vidhya - Health Insurance\test_processed.csv'
SUBMISSION_FILE_PATH = r'C:\Users\vishw\Desktop\Job Prep\Analytics Vidhya - Health Insurance'

def load_data(path):
    '''
    
        Function to load data, in this case a csv file
        
    '''
    return pd.read_csv(path)

def save_data(data,path):
    '''
    
        Function to save data, in this case a csv file
      
        Inputs:
            1. data - Dataframe
            2. path - string
    '''
    return data.to_csv(path,index = False)

def prepare_data(train_df_raw,test_df_raw,data_prep_dict):
    '''
    
        Function to process raw data into required modelling data
        
        Inputs:
            1. train_df_raw - Dataframe
            2. test_df_raw  - Dataframe
            3. data_prep_dict - Dictionary
        
        Outputs:
            1. train_df_processed - Dataframe
            2. test_df_processed - Dataframe
    '''

    #quick check to apply data processing on both train and test combined
    #train_df_raw = pd.concat([train_df_raw,test_df_raw],axis = 0)
    
    #override simple imputer error by manually assigning missing values
    train_df_raw['Holding_Policy_Duration'].fillna('-1',inplace=True)
    test_df_raw['Holding_Policy_Duration'].fillna('-1',inplace=True)
    train_df_raw.fillna('missing',inplace = True)
    test_df_raw.fillna('missing',inplace = True)
    
    #modify data values to convert catergorical raw attributes to potential numeric features
    
    train_df_raw.replace({'14+':'14'},inplace = True)
    train_df_raw['Holding_Policy_Duration'] = train_df_raw['Holding_Policy_Duration'].astype(float)
    test_df_raw.replace({'14+':'14'},inplace = True)
    test_df_raw['Holding_Policy_Duration'] = test_df_raw['Holding_Policy_Duration'].astype(float)
    
    #freeze data types
    train_df_raw[data_prep_dict['one_hot_encode']] = train_df_raw[data_prep_dict['one_hot_encode']].astype(str)
    test_df_raw[data_prep_dict['one_hot_encode']] = test_df_raw[data_prep_dict['one_hot_encode']].astype(str)
    
    #target encode required attributes
    for target_encode_col in data_prep_dict['target_encode']:
        encoding_dict = train_df_raw.groupby(target_encode_col)[TARGET].mean().to_dict()
        train_df_raw[target_encode_col] = train_df_raw[target_encode_col].map(encoding_dict)
        test_df_raw[target_encode_col] = test_df_raw[target_encode_col].map(encoding_dict)
    
    #fill missing Region Codes
    #city_code_means = train_df_raw.groupby(['City_Code'])[TARGET].mean().reset_index()
    #test_df_raw['Region_Code'] = test_df_raw.apply(
    #lambda row: city_code_means[TARGET][city_code_means.City_Code ==
    #                                    row['City_Code']].values[0]
    #                                if row['Region_Code'] not in train_df_raw['Region_Code'].unique() else row['Region_Code'],
    #                            axis=1
    #                        )
    
    #define set of transformation steps per raw attribute present in the data

    column_transformer_1 = ColumnTransformer(
        [
         ('one_hot_encode', OneHotEncoder(sparse = False,drop = 'if_binary'), data_prep_dict['one_hot_encode'])
        ],
        remainder = 'passthrough',
        verbose = 'True')

    #build and fit the column transformer on train data
    train_df_processed = column_transformer_1.fit_transform(train_df_raw)
    #apply the column transformer on test data
    test_df_processed = column_transformer_1.transform(test_df_raw)
    
    #convert numpy arrays into pandas dataframe for further analysis
    train_df_processed_1 = pd.DataFrame(train_df_processed,columns = column_transformer_1.get_feature_names())
    test_df_processed_1 = pd.DataFrame(test_df_processed,columns = column_transformer_1.get_feature_names())
    
    column_transformer_2 = ColumnTransformer(
        [('passthrough','passthrough',[col for col in
                                       train_df_processed_1.columns if col not
                                       in data_prep_dict['standard_scale']]),
         ('standard_scale', StandardScaler(), data_prep_dict['standard_scale'])
        ],
        remainder = 'passthrough',
        verbose = 'True')
    
    #build and fit the column transformer on train data
    train_df_processed_2 = column_transformer_2.fit_transform(train_df_processed_1)
    #apply the column transformer on test data
    test_df_processed_2 = column_transformer_2.transform(test_df_processed_1)    
    
    #recreate column names in the correct order, to understand feature importances
    train_df_processed_out = pd.DataFrame(train_df_processed_2,columns = [col for col in
                                       train_df_processed_1.columns if col not
                                       in data_prep_dict['standard_scale']] + data_prep_dict['standard_scale'])
    test_df_processed_out = pd.DataFrame(test_df_processed_2,columns = [col for col in
                                       train_df_processed_1.columns if col not
                                       in data_prep_dict['standard_scale']]+ data_prep_dict['standard_scale'])
    
    #progress logger
    print('Target encoding completed, return processed data')    
    
    return train_df_processed_out, test_df_processed_out   

def train_model(modelling_data,features):
    '''
    
        Function to train a model using XGBoost
        
        Inputs:
            1. modelling_data - Dataframe
            2. features  - list of strings
        
        Outputs:
            1. trained_model - Xgboostmodel
    '''    
    parameters = {'nthread':[4], 
              'objective':['binary:logistic'],
              'eval_metric': ['logloss'],
              'learning_rate': [0.05], 
              'max_depth': [6],
              'min_child_weight': [9,10,11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.5],
              'n_estimators': [100],
              'missing':[-999],
              'seed': [1337]}
    
    xgb_model = xgb.XGBClassifier(use_label_encoder = False)
    
    clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv = 10, 
                   scoring = 'roc_auc',
                   verbose=2, refit=True)
    
    clf.fit(modelling_data[features], modelling_data[TARGET])
    
    print('AUC SCORE ----> ',clf.best_score_)
    print(clf.best_params_)
    
    return clf

def train_model_catboost(modelling_data,features,categorical_features):
    '''
    
        Function to train a model using XGBoost
        
        Inputs:
            1. modelling_data - Dataframe
            2. features  - list of strings
        
        Outputs:
            1. trained_model - Xgboostmodel
    '''    

    parameters = {'iterations': [500],
              'depth': [4, 5, 6],
              'loss_function': ['Logloss'],
              'l2_leaf_reg': np.logspace(-20, -19, 3),
              'leaf_estimation_iterations': [10],
    #           'eval_metric': ['Accuracy'],
    #           'use_best_model': ['True'],
              'logging_level':['Silent'],
                'random_seed': [42]
         }
    
    model = CatBoostClassifier()
    
    clf = GridSearchCV(model, parameters, n_jobs=5, 
                   cv = 10, 
                   scoring = 'accuracy',
                   verbose=2, refit=True)
    
    clf.fit(modelling_data[features], modelling_data[TARGET], categorical_features)
    
    print('AUC SCORE ----> ',clf.best_score_)
    print(clf.best_params_)
    
    return clf

def explore_algorithms(modelling_data,features):
    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'roc_auc'
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=10)
    	cv_results = model_selection.cross_val_score(model, modelling_data[features], modelling_data[TARGET], cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)

def prepare_train_validation_data(modelling_data):
    '''
    
        Function to generate a validation data set from training data 
        
        Inputs:
            1. modelling_data - Dataframe
        
        Outputs:
            1. train_df - Dataframe
            2. validation_df - Dataframe
            
    '''    
    modelling_data.sample(frac = 1)
    validation_df = modelling_data.tail(5000)
    train_df = modelling_data.head(modelling_data.shape[0] - 5000)
    
    return train_df, validation_df
    
def evaluate(data,label,features,model):
    '''
    
        Function to evaluate model performance on hold out data 
        
        Inputs:
            1. data - Dataframe
            2. label - string
            3. features - list of strings
            3. model - classifier
        
        Outputs:
            
            1. score - float
            
    ''' 
    print('Performance on Holdout Data ---->',
          roc_auc_score(data[TARGET], model.predict_proba(data[features])[:, 1])) 
    
    return roc_auc_score(data[TARGET], model.predict_proba(data[features])[:, 1])

def create_submission(data,features,model, iteration_id):
    '''
    
        Function to evaluate model performance on hold out data 
        
        Inputs:
            1. data - Dataframe
            2. features - list of strings
            3. model - classifier
            4. iteration_id - string
        
        Outputs:
            
            4. preds - Dataframe
            
    '''     
    data[TARGET] = model.predict_proba(data[features])[:,1]
    data[PRIMARY_KEY] = data[PRIMARY_KEY].astype(int)
    data[PRIMARY_KEY] = data[PRIMARY_KEY].astype(str)
    data[[PRIMARY_KEY,TARGET]].to_csv(SUBMISSION_FILE_PATH+'\submission_{0}.csv'.format(iteration_id),index = False)
    
    return data

def test_prediction(train,test,features):
    """Try to classify train/test samples from total dataframe"""

    train['target'] = 1
    test['target'] = 0    
    combined = pd.concat([train[features+['target']],test[features+['target']]],axis = 0)
    print(combined.shape)
    # Create a target which is 1 for training rows, 0 for test rows

    # Perform shuffled CV predictions of train/test label
    predictions = cross_val_predict(
        ExtraTreesClassifier(n_estimators=100, n_jobs=4),
        combined[[col for col in combined.columns if col not in [PRIMARY_KEY,TARGET,'target']]], combined['target'],
        cv=StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=42
        )
    )

    # Show the classification report
    print(classification_report( combined['target'], predictions))
    
