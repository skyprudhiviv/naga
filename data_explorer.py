"""
                Analytics Vidhya Jobathon

File Description: Data Cleaning + Exploration for the Health Insurance prediction challenge
Date: 27/02/2021                
Author: vishwanath.prudhivi@gmail.com
"""

#import required libraries
from pandas_profiling import ProfileReport


#import user created libraries
from utils import load_data,prepare_data,save_data,test_prediction,TARGET,\
                  RAW_TRAIN_DATA_PATH,RAW_TEST_DATA_PATH,\
                  PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH

#load the datasets
train_df = load_data(RAW_TRAIN_DATA_PATH)
test_df = load_data(RAW_TEST_DATA_PATH)

#visualize raw data using a pandas profiler
#train_profile = ProfileReport(train_df, title="Pandas Profiling Report")
#train_profile.to_file(r'C:\Users\vishw\Desktop\Job Prep\Analytics Vidhya - Health Insurance\train_df.html')

#test_profile = ProfileReport(test_df, title="Pandas Profiling Report")
#test_profile.to_file(r'C:\Users\vishw\Desktop\Job Prep\Analytics Vidhya - Health Insurance\test_df.html')

#feature engineering section
#create a dummy target column in test data
test_df[TARGET] = 2
train_df['age_diff'] = train_df['Upper_Age'] - train_df['Lower_Age']
test_df['age_diff'] = test_df['Upper_Age'] - test_df['Lower_Age']

train_df[['age_diff','Upper_Age','Lower_Age','Response']].corr()


#define set of data cleaning + processing steps required

data_prep = {
    'impute': ['Health Indicator','Holding_Policy_Duration','Holding_Policy_Type'],
    'one_hot_encode': ['Accomodation_Type','Reco_Insurance_Type',
                      'Health Indicator','City_Code', 'Is_Spouse','Reco_Policy_Cat'],
    'standard_scale': ['Reco_Policy_Premium','Upper_Age','Lower_Age','age_diff','Region_Code'],
    'target_encode': ['Region_Code'],
    'passthrough': [TARGET]
    }

#call data preparation module
train_df_processed, test_df_processed = prepare_data(train_df,test_df,data_prep)


#save date to disk
save_data(train_df_processed,PROCESSED_TRAIN_DATA_PATH)
save_data(test_df_processed,PROCESSED_TEST_DATA_PATH)

