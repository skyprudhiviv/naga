"""
                Analytics Vidhya Jobathon

File Description: Model Build for the Health Insurance prediction challenge
Date: 27/02/2021                
Author: vishwanath.prudhivi@gmail.com
"""

#import required libraries
import pandas as pd
import numpy as np
from datetime import datetime
#import user created libraries
from utils import load_data,prepare_data,save_data,prepare_train_validation_data,\
                  train_model,train_model_catboost,evaluate,create_submission,test_prediction,explore_algorithms,TARGET,PRIMARY_KEY,\
                  PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,SUBMISSION_FILE_PATH,RAW_TRAIN_DATA_PATH,RAW_TEST_DATA_PATH

#set the seed for reproducability
np.random.seed(100)

#load the datasets
modelling_df = load_data(PROCESSED_TRAIN_DATA_PATH)
test_df = load_data(PROCESSED_TEST_DATA_PATH)

#select feature set
EXCLUDE = ['Blank','Holding_Policy_Type','Holding_Policy_Duration']
FEATURES = [col for col in modelling_df.columns if col not in [TARGET,PRIMARY_KEY] + EXCLUDE]

    
#prepare train + validation data
train_df , validation_df = prepare_train_validation_data(modelling_df)

#fit the model using cross validation grid search
model = train_model(train_df,FEATURES)

#check model performance on holdout data
score = evaluate(validation_df, TARGET, FEATURES,model)

#driver analysis
importances = pd.DataFrame({'feature':FEATURES,'importance':model.best_estimator_.feature_importances_})

#generate predictions on test data
#out = create_submission(test_df,FEATURES,model,str(score) +'_' + str(datetime.now().strftime("%Y%m%d-%H%M%S")))