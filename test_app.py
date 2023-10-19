from Pred_File_Data_Validation.predFileDataValidation import pred_validation_functions
from Logs_Writer.logger import App_Logger

log_writer = App_Logger()
file_object = open("Pred_Logs/Test_logs.txt", 'a+')

### Test ###
uploaded_file_name = 'InputFile'
file_path = 'Pred_Uploaded_File/uploaded_file.csv'
log_writer.log(file_object,'Starting Validation on files for Uploaded File.')
test_pred = pred_validation_functions(file_path, uploaded_file_name)
test_pred.validateFileType()
NumberofColumns , column_names = test_pred.load_schema_values()     # Extracting values from prediction schema
test_pred.validateTableStructure(NumberofColumns, column_names)     # Checking the number of columns and column names in the uploaded file
log_writer.log(file_object,'Validation on files is completed.')     # and Inserting the validated file in Pred_Valid_File
