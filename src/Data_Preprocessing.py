# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler 
from Logs_Writer.logger import App_Logger

         

class Process_train_data:
    def __init__(self):
        self.file = open("Train_Logs/Data_Preprocessing_Logs.log", 'w+') 
        self.log_writer = App_Logger()

    def drop_unwanted_features(self,df):
        '''Drops Unwanted Columns'''
        df.drop(columns=['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured','TBG','TSH'],inplace=True)
        self.log_writer.log(self.file , "Unwanted columns have been droped.")
        return df


    def replace_missing_vals_with_nulls(self,df):
        df.replace({"?":np.nan}, inplace=True)
        self.log_writer.log(self.file, "All missing values('?') are replaced into Null values(np.nan)")
        return df


    def label_encode(self, df):
        ''' Encodes all the Categorical Values into proper numeric values. 
            And saves the label encoder for further Prediction
            Returns a X: Features and y: Target variable "Class"
        '''
        
        # We can map the categorical values like below:
        df['sex'] = df['sex'].map({'F': 0, 'M': 1})
        # Except for 'sex' column all the other columns with two categorical data have same value 'f' and 't'
        # so instead of mapping indvidually, let's create a loop
        for column in df.columns:
            if len(df[column].unique()) == 2:
                df[column] = df[column].map({'f': 0, 't': 1})

        # Now let's deal with column with more than 2 categories, by using get_dummies
        df = pd.get_dummies(df,columns=['referral_source'], dtype=float, drop_first=True )
        le = LabelEncoder()
        df['Class'] = le.fit_transform(df['Class'])
        self.log_writer.log(self.file, 'Categorical columns have been Label Encoded')

        with open('src/Saved_models/le.pickle', 'wb') as file:
            pickle.dump(le, file)
        self.log_writer.log(self.file, 'Encoder fitted on data is saved in  Saved_models file.')

        # Separating Features and Labels
        X = df.drop(columns=['Class'])
        y = df[['Class']]
        self.log_writer.log(self.file, 'Features and Label is separated.')

        return X,y


    def fill_missing_values(self, feature_data):
        """ Method Name: impute_missing_values
            Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
        """
        self.log_writer.log(self.file, 'Entered the impute_missing_values method of the Preprocessor class')
            
        try:
            imputer = KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            new_array = imputer.fit_transform(feature_data) # impute the missing values
            
            # convert the new_array returned to a Dataframe
            # rounding the value because KNNimputer returns value between 0 and 1, but we need either 0 or 1
            no_null_data = pd.DataFrame(data=np.round(new_array), columns=feature_data.columns) 
            self.log_writer.log(self.file, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')

        except Exception as e:
            self.log_writer.log(self.file, 'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file, 'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

        return no_null_data


    def set_imbalance_off(self,X,y):
        ''' Uses RandomOverSampler to handle the Imbalance in the dataset by oversampling.
            And sets a balance in all the categories in y target variabale.
        '''
        rdsmple = RandomOverSampler()
        x_balanced, y_balanced = rdsmple.fit_resample(X, y)
        # Converting the balanced_arrays into a Dataframes
        X = pd.DataFrame(data = x_balanced, columns = X.columns)
        y = pd.DataFrame(data = y_balanced, columns = ['Class'])

        processed_df = pd.concat([X, y], axis=1)
        processed_df.to_csv("data_sets/processed_df.csv", index=False, header=True)
        self.log_writer.log(self.file, 'Successfully saved the processed and balanced Data in a csv file in data_sets directory.')

        return X, y
