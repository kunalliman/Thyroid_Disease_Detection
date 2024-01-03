import os
import pickle
from Logs_Writer.logger import App_Logger


class modelFinder():
    def __init__(self):
        self.log_writer = App_Logger()
        self.file_object = open('Pred_Logs\Pred_Model_Finder_Logs.txt','w+')        

    def model_for_cluster(self, cluster):
        self.log_writer.log(self.file_object, f'Finding Model for Cluster {cluster}.')
        try:
            for model in os.listdir('src/Saved_models'):
                    if cluster in model:
                        with open("src/Saved_models/" + f"{model}",'rb') as f:
                            model = pickle.load(f)
        
        except Exception as e:
             self.log_writer.log(self.file_object, f'Error Occured while Finding Model for Cluster {cluster}.')
             self.log_writer.log(self.file_object, f'Error: {e}')
             raise e
        
        self.log_writer.log(self.file_object, f'{model} Model for Cluster {cluster} is returned.')
        return model
