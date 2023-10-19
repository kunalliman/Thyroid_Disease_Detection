import pandas as pd
from Logs_Writer.logger import App_Logger

        

class Ingest_train_data:
    def __init__(self):
        self.file = open("Train_Logs/Data_Ingestion_Logs.log", 'a+') 
        self.log_writer = App_Logger() 

    def data_validation(self, csv_file_path):
        ''' Method Name: data_validation 
            Description: To check the provided CSV have the expected columns.
        '''
        NumberofColumns = 30
        column_list=['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant',
                 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre',
                 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'TSH',  'T3_measured', 'T3', 'TT4_measured', 'TT4',
                 'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG', 'referral_source', 'Class']
        try:
            self.log_writer.log(self.file, "Reading the file...")
            df = pd.read_csv(csv_file_path)
            self.log_writer.log(self.file, "CSV file read successfully.")

            self.log_writer.log(self.file, "Validating Data in the provided CSV File")
            if df.shape[1] == NumberofColumns:
                if list(df.columns) == column_list:
                    df.to_csv("data_sets/validated_df.csv", index=False, header=True)
                self.log_writer.log(self.file, "Dataset is validated.")
                self.log_writer.log(self.file," Valid Dataset is Ingested successfully in data_sets as 'validated_df.csv' ")

            else:
                self.log_writer.log(self.file, "Provided data is Invalid")
                raise ValueError("Invalid File: The provided CSV file does not have the expected columns.")
            
        except Exception as e:
            self.log_writer.log(self.file, f"An error occurred during data validation: {str(e)}")
            raise e
        
        return df
        





