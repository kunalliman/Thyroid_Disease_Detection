# Importing the necessary libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import KNNImputer
from Logs_Writer.logger import App_Logger

class preprocessor:
    def __init__(self):
        self.log_writer = App_Logger()
        self.file_object = open('Pred_Logs\Pred_Data_Preprocessor_Logs.txt','w+')

    def drop_unwanted_cols(self, df):
        # Now let's remove the unwanted columns as discussed in the EDA part in ipynb file.
        df.drop(columns=['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured','TBG','TSH'],inplace=True)
        self.log_writer.log(self.file_object, "Unwanted columns have been droped.")
        return df

    def replacing_missing_values(self, df):
        # We saw that the missing values were imputed by a "?". Let's convert them into null value.
        df.replace({"?":np.nan}, inplace=True)
        self.log_writer.log(self.file_object, "All missing values('?') are replaced into Null values(np.nan)")
        return df


    def category_encoding(self, df):
        try:
            # LabelEncoding categorical data
            # We can map the categorical values like below:
            df['sex'] = df['sex'].map({'F': 0, 'M': 1})

            # Except for 'sex' column all the other columns with two categorical data have same value 'f' and 't'.
            # so instead of mapping indvidually, let's create a loop
            cat_data = df.drop(['age','T3','TT4','T4U','FTI','sex'],axis=1)
            for column in cat_data.columns:
                if ('f' and 't') in df[column].str.lower().unique():
                    df[column] = df[column].str.lower().map({'f': 0, 't': 1})

                elif 'f' in df[column].str.lower().unique():   # For columns with only False values
                    df[column] = df[column].str.lower().map({'f': 0})  

            # Now let's deal with column with more than 2 categories, by using get_dummies.
            df = pd.get_dummies(df,columns=['referral_source'], dtype=float, drop_first=True )
            self.log_writer.log(self.file_object, 'Categorical columns have been Label Encoded')

        except Exception as e:
            self.log_writer.log(self.file_object, f"Error Ocurred while Encoding Category columns: {e} ")
            raise e 
        
        return df


    def impute_missing_values(self, df):
        """ Method Name: impute_missing_values
            Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
        """
        self.log_writer.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        
        try:
            imputer = KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            new_array = imputer.fit_transform(df) # impute the missing values
            # convert the new_array returned in the step above to a Dataframe
            # rounding the value because KNNimputer returns value between 0 and 1, but we need either 0 or 1
            new_data = pd.DataFrame(data=np.round(new_array), columns= df.columns)
            self.log_writer.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
    
        except Exception as e:
            self.log_writer.log(self.file_object, 'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object, 'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise e
        return new_data
    
    def create_clusters(self,df):
        with open('src\Saved_models\KMeans.pickle','rb') as f:
            kmeans_model = pickle.load(f)

        clusters = kmeans_model.predict(df)
        df['Cluster'] = clusters
        clusters_formed = df['Cluster'].unique()
        return df, clusters_formed








