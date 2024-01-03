import json
import re
import os
import pandas as pd
from Logs_Writer.logger import App_Logger


class pred_validation_functions:

    def __init__(self, file_path, uploaded_file_name):
        self.file_path = file_path
        self.uploaded_file_name = uploaded_file_name
        self.schema_path = 'schema_prediction.json'
        self.NumberofColumns, self.column_names = self.load_schema_values()
        self.file = open("Pred_Logs/Pred_Data_validation_Logs.txt", 'w+') 
        self.log_writer = App_Logger()   

    def validateFileType(self):
        valid_extension = r"\.csv$"
        try:
            self.log_writer.log(self.file, f"Validating File Type with regex '{valid_extension}' ")
            if re.search(valid_extension, self.uploaded_file_name, re.IGNORECASE):
                self.log_writer.log(self.file, f'Uploaded file {self.uploaded_file_name} has valid file type.')
            else:
                self.log_writer.log(self.file, f'Uploaded file {self.uploaded_file_name} has Invalid file type.')
                self.clean_directory(directory_path='Pred_Uploaded_File')
                self.log_writer.log(self.file, f"Uploaded file is removed from directory 'Pred_Uploaded_File' ")

        except ValueError:
            self.log_writer.log(self.file, f"Error Occured while Validating File : {ValueError}")
            raise ValueError
        except OSError:
            self.log_writer.log(self.file, f"Error Occured while Deleting File : {OSError}")
            raise OSError
        except Exception as e:
            self.log_writer.log(self.file, f"Error Occured : {e}")
            raise e

    def load_schema_values(self):
        ''' Method Name: load_schema_values
            Description: This method extracts reuired information from the pre-defined "Schema" file.
        '''
        try:
            with open(self.schema_path, "r") as schema_file:
                schema_dict = json.load(schema_file)
                NumberofColumns = schema_dict.get("NumberofColumns")
                column_names = schema_dict.get("ColName")
                # self.log_writer.log(self.file, f'Number of columns:{NumberofColumns} , and column names : {column_names} are extracted from schema file {schema_file}.')
                return NumberofColumns , column_names
            
        except ValueError:
            # self.log_writer.log(self.file, f"Error Occured while extracting the values from Schema File : {ValueError}")
            raise ValueError    
        except Exception as e:
            # self.log_writer.log(self.file, f"Error Occured : {e}")
            raise e

    def validateTableStructure(self, NumberofColumns, column_names):
        # Read the CSV data into a DataFrame
        df = pd.read_csv(self.file_path)
        file_name = self.file_path.split('/')[1]
        try:
            if df.shape[1] == NumberofColumns:
                if ( df.columns == list(column_names.keys()) ).all():
                    df.to_csv("Pred_Validated_File/" + file_name , index=None, header=True)
                else:pass
                    # self.log_writer.log(self.file, f"Column names do not match the expected structure. Uploaded file column list: {list(column_names.keys())}")
            else:pass
                # self.log_writer.log(self.file, f"Number of columns does not match the expected structure. There may be a column missing. Uploaded file column length: {NumberofColumns}")

        except ValueError:
            # self.log_writer.log(self.file, f"Error Occured while Validating Table Structure : {ValueError}")
            raise ValueError
        except Exception as e:
            # self.log_writer.log(self.file, f'Error Occured : {e}')
            raise e
        
    def clean_directory(self, directory_path):
        try:
            # self.log_writer.log(self.file, f'Cleaning Files from directory {directory_path}')
            for file in os.listdir(directory_path):
                os.remove(directory_path + '\\' + file)
                # self.log_writer.log(self.file, f'File {file} removed from directory {directory_path}')
        except OSError as e:
            self.log_writer.log(self.file, f'Error Occured while Cleaning Files from directory: {e}')
            raise e
        


