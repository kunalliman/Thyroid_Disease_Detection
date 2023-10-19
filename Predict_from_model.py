import os
import pandas as pd
import pickle
from Pred_Data_Preprocessor.predDataPreprocessor import preprocessor
from Pred_Model_Finder.predModelFinder import modelFinder
from Logs_Writer.logger import App_Logger




class prediction:
    def __init__(self, uploaded_file_name):
        self.uploaded_file_name = uploaded_file_name
        self.file_path = f'Pred_Valid_File\{uploaded_file_name}'
        self.log_writer = App_Logger()
        self.file_object = open('Pred_Logs\Prediction_From_Model_Logs', 'a+')

    def predictionFromModel(self):
        try:
            pred_df = pd.read_csv(f'Pred_Validated_File/{self.uploaded_file_name}')
            self.log_writer.log(self.file_object, 'Preprocessing process started.')
            processor = preprocessor()

            # Dropping unwanted Columns
            pred_df = processor.drop_unwanted_cols(pred_df)

            # Replacing '?' with np.nan values
            pred_df = processor.replacing_missing_values(pred_df)

            # Encoding categoricla columns
            pred_df = processor.category_encoding(pred_df)

            # Imputing null values if present any
            if pred_df.isna().sum()[pred_df.isna().sum()>0].empty :
                pass 
            else: pred_df = processor.impute_missing_values(pred_df)

            self.log_writer.log(self.file_object, 'Preprocessing process completed.')


            self.log_writer.log(self.file_object, 'Clustering process started.')
            # Create clusters for runnig a model on each for better prediction
            df, clusters_formed = processor.create_clusters(pred_df)

            # Encoder for inverse encoding of the predictions to Class names
            with open('src/Saved_models/le.pickle', 'rb') as file: 
                encoder = pickle.load(file)

            self.log_writer.log(self.file_object, 'Clusters are formed in prediction data and encoder has loaded for decoding the predictions.')

            self.log_writer.log(self.file_object, 'Prediction started for each cluster.')
            for c in clusters_formed:
                cluster_df = df.loc[df['Cluster']== c]
                cluster_df.drop(columns=['Cluster'], inplace=True)
                finder = modelFinder()
                self.log_writer.log(self.file_object, f'Finding model for cluster {c} with the help of model_for_cluster function from.')
                model = finder.model_for_cluster(cluster= 'c')    
                predictions = model.predict(cluster_df)
                predicted_classes = encoder.inverse_transform(predictions)  # Decoding predictions
                cluster_df['Predicted_Class'] = predicted_classes
                cluster_df.to_csv(f'PREDICTIONs/Cluster_{c}_predictions.csv', index=False, header=True)
            self.log_writer.log(self.file_object, 'Predictions for each custer is stored in "PREDICTIONs".')

        except Exception as e:
            raise e
        return print('Code Completed!')
    
    def save_main_result(self):
        result_df = pd.DataFrame()
        predictions_directory = 'PREDICTIONs'  # Directory where prediction files are located
        
        for pred_file in os.listdir(predictions_directory):
            pred_file_path = os.path.join(predictions_directory, pred_file)
            
            if pred_file.endswith('predictions.csv'):
                pred_df = pd.read_csv(pred_file_path)  # Load the prediction data from the file
                result_df = pd.concat([result_df, pred_df], axis=1)

        result_df.to_csv('PREDICTIONs/Predicted_results.csv', index=False)  # Save the concatenated DataFrame to a CSV file


        




    
        