from Pred_File_Data_Validation.predFileDataValidation import pred_validation_functions 
from Logs_Writer import logger


class pred_data_validation_and_insertion:   ### This class consists steps to validate data using the functions written in  pred_raw_data_valiation_functions Class
    def __init__(self,file_path, uploaded_file_name):
        self.file_data = pred_validation_functions(file_path, uploaded_file_name)
        self.file_object = open("Pred_Logs/Pred_File_Validtaion_Logs.txt", 'w+')
        self.log_writer = logger.App_Logger()

    def pred_data_validate_and_insert(self):

        try:
            
            self.log_writer.log(self.file_object,'Starting Validation on files for Uploaded File.')

            # Checking file type
            self.file_data.validateFileType()

            # Extracting values from prediction schema
            NumberofColumns , column_names = self.file_data.load_schema_values()
            
            # Checking the number of columns and column names in the uploaded file and Inserting the validated file in Pred_Valid_File
            self.file_data.validateTableStructure(NumberofColumns, column_names)
            
            self.log_writer.log(self.file_object,"File and Data Validated, and Insertion Completed.")
            #*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--

           
        except Exception as e:
            raise e

from Predict_from_model import prediction 


### Main Testing ###

# uploaded_file_name = 'InputFile.csv'
# file_path = 'Pred_Uploaded_File/InputFile.csv'
# pred_valid_insert = pred_data_validation_and_insertion(file_path ,uploaded_file_name)
# pred_valid_insert.pred_data_validate_and_insert()
# pred = prediction(uploaded_file_name)
# pred.predictionFromModel() # Will create prediction files for each cluster
# pred.save_pred_results('Pred_Uploaded_File/InputFile.csv')    
# # Saves the result from the predicted files to the uploaded file and save it by as 'Predicted_results.csv' in PREDICTIONs
                                                            






