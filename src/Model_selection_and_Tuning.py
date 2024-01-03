import pandas as pd
import logging
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from Logs_Writer.logger import App_Logger

        
class Model_fitter:
    def __init__(self):
        self.file = open("Train_Logs/Model_selection_and_Tuning_Logs.log", 'w+') 
        self.log_writer = App_Logger() 

    def get_best_params_for_random_forest(self,train_x,train_y):
        """ Method Name: get_best_params_for_random_forest
            Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
            Output: Return RandomForestClassifier model with the best parameters
        """
        self.log_writer.log(self.file, 'Entered the get_best_params_for_random_forest.')
        try:
            # initializing with different combination of parameters
            param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                            "max_depth": range(2, 4, 1), "max_features": [None, 'sqrt', 'log2']}

            #Creating an object of the Grid Search class
            grid_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5,  verbose=3)
            grid_rfc.fit(train_x, train_y) #finding the best parameters

            #extracting the best parameters
            criterion = grid_rfc.best_params_['criterion']
            max_depth = grid_rfc.best_params_['max_depth']
            max_features = grid_rfc.best_params_['max_features']
            n_estimators = grid_rfc.best_params_['n_estimators']

            #creating a new model with the best parameters
            rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                            max_depth=max_depth, max_features=max_features)
            # training the mew model
            rfc.fit(train_x, train_y)
            self.log_writer.log(self.file, 'Random Forest best params: '+ str(grid_rfc.best_params_)+' are returned.')
        
        except Exception as e:
            self.log_writer.log(self.file, 'Exception occured in get_best_params_for_random_forest method. Exception message:  ' +str(e))
            self.log_writer.log(self.file, 'Random Forest Parameter tuning failed. Exited the get_best_params_for_random_forest method.')
            raise e
        
        return rfc    #### RandForestClassifier Model with best parameters
    

    def get_best_params_for_KNN(self,train_x, train_y):
        """ Method Name: get_best_params_for_KNN
            Description: get the parameters for KNN Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
            Output: Return KNeighborsClassifier model with the best parameters

        """
        self.log_writer.log(self.file, 'Entered the get_best_params_for_KNN method.')
        try:
            # initializing with different combination of parameters
            param_grid_knn = {
                'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size' : [10,17,24,28,30,35],
                'n_neighbors':[4,5,8,10,11],
                'p':[1,2]
            }

            # Creating an object of the Grid Search class
            grid_knn = GridSearchCV(estimator= KNeighborsClassifier(), param_grid = param_grid_knn, verbose=3,cv=5)
            grid_knn.fit(train_x, train_y) # finding the best parameters

            # extracting the best parameters
            algorithm = grid_knn.best_params_['algorithm']
            leaf_size = grid_knn.best_params_['leaf_size']
            n_neighbors = grid_knn.best_params_['n_neighbors']
            p  = grid_knn.best_params_['p']

            # creating a new model with the best parameters
            knn = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size, n_neighbors=n_neighbors,p=p,n_jobs=-1)
            # training the mew model
            knn.fit(train_x, train_y)
            self.log_writer.log(self.file, 'KNN best params: ' + str(grid_knn.best_params_) + ' are returned.')
        
        except Exception as e:
            self.log_writer.log(self.file, f'Exception occured in knn method. Exception message: ' + str(e))
            self.log_writer.log(self.file, 'knn Parameter tuning  failed. Exited the knn method.')
            raise e
        
        return knn   #### KNeighborsClassifier Model with best parameters

    ### This mthod will extract the RFC and KNN models from the above two functions and compare return the Best Model 
    def get_best_model(self, train_x,train_y,test_x,test_y):
        """ Method Name: get_best_model
            Description: Find out the Model which has the best AUC score.
            Output: The best model name and the model object
        """
        self.log_writer.log(self.file, 'Entered the get_best_model method.')
        
        try:
            # create best model for KNN
            knn = self.get_best_params_for_KNN(train_x,train_y)
            prediction_knn = knn.predict_proba(test_x) # Predictions using the KNN Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                knn_score = accuracy_score(test_y, prediction_knn)
                self.log_writer.log(self.file, 'Accuracy for knn:' + str(knn_score))  # Log AUC
            else:
                knn_score = roc_auc_score(test_y, prediction_knn, multi_class='ovr') # AUC for KNN
                self.log_writer.log(self.file, 'AUC for knn:' + str(knn_score)) # Log AUC


            # create best model for Random Forest
            random_forest= self.get_best_params_for_random_forest(train_x,train_y)
            prediction_random_forest=random_forest.predict_proba(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                random_forest_score = accuracy_score((test_y),prediction_random_forest)
                self.log_writer.log(self.file, 'Accuracy for RF: ' + str(random_forest_score))
            else:
                random_forest_score = roc_auc_score((test_y), prediction_random_forest,multi_class='ovr') # AUC for Random Forest
                self.log_writer.log(self.file, 'AUC for RF: ' + str(random_forest_score))

            ####### Comparing the two models and returning (Best_Model_Name, Best_Model) 
            if(random_forest_score <  knn_score):
                return 'KNN', knn            
            else: 
                return 'RandomForest',random_forest   

        except Exception as e:
            self.log_writer.log(self.file, 'Exception occured in get_best_model method. Exception message:  ' + str(e))
            self.log_writer.log(self.file, 'Model Selection Failed. Exited the get_best_model method.')
            raise e
        






     