import logging
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from src.Data_Ingestion import Ingest_train_data
from src.Data_Preprocessing import Process_train_data
from src.Clustering import form_clusters
from src.Model_selection_and_Tuning import Model_fitter


logging.basicConfig(filename="Train_Logs/Main_Logs.log", level=logging.INFO, format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


def main():

    logging.info("Started with Validating of data")
    train_file_path = "data_sets\hypothyroid.csv"
    ingest = Ingest_train_data()
    valid_df = ingest.data_validation(train_file_path)   # VAlidate the number of features and feature names in the csv and returns the validated DataFrame
    logging.info("Successfully Validated data.")


    logging.info("Started with Preprocessing of data")
    processor = Process_train_data()
    df = processor.drop_unwanted_features(valid_df)             # First let's remove the unwanted columns as discussed in the EDA part in ipynb file
    df = processor.replace_missing_vals_with_nulls(df)          # We saw that the missing values were imputed by a "?". Let's convert them into null value
    X_features, y_target_class = processor.label_encode(df)     # Label encoding the categorical Values with proper stratergy
    X_features = processor.fill_missing_values(X_features)      # Method for Imputing nulls
    X, y = processor.set_imbalance_off(X_features, y_target_class)   # Balnacing data for better Training of model
    logging.info("Successfully Preprocessed the data.")


    logging.info("Started with Clustering of data")
    cluster_maker = form_clusters()
    optimum_clusters = cluster_maker.elbow_plot(X)                              # Optimum number of Clusters to form
    X = cluster_maker.create_clusters(data=X, k_clusters = optimum_clusters)    # A column named Cluster will be added in X Features 
    clustered_df = pd.concat([X, y], axis=1)
    list_of_clusters = clustered_df['Cluster'].unique()
    clustered_df.to_csv("data_sets/clustered_df.csv", index=False)
    logging.info('Successfully saved the data with cluster information in a csv file in data_sets directory.')


    logging.info("Started with Model Selection for each Cluster formed in the data")
    model_trainer = Model_fitter()
    # Segregating the data for each cluster using for loop

    for i in list_of_clusters:
        cluster_data = clustered_df.loc[clustered_df['Cluster']==i] # filter the data for one cluster
        # Prepare the feature and Label columns
        cluster_features = cluster_data.drop(['Class','Cluster'],axis=1)
        cluster_label = cluster_data['Class']

        # splitting the data into training and test set for each cluster one by one
        x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1/3, random_state=355)

        #getting the best model for each of the clusters
        best_model_name, best_model = model_trainer.get_best_model(x_train,y_train,x_test,y_test)

        #saving the best model to the directory with Cluster_Number(i)
        with open(f'src/Saved_models/{best_model_name+str(i)}.pickle', 'wb') as file: 
            pickle.dump(best_model, file)     
            logging.info(f'For Cluster {i}, Best Model is {best_model_name+str(i)}. Model is saved in Saved_models file.')


if __name__ == "__main__":
    main()