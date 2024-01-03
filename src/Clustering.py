import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from kneed import KneeLocator
from Logs_Writer.logger import App_Logger


class form_clusters:
    def __init__(self):
        self.file = open("Train_Logs/Clustering_Logs.log", 'w+') 
        self.log_writer = App_Logger()       


    def elbow_plot(self, data):
        """ Method Name: elbow_plot
            Description: This method saves the plot to decide the optimum number of clusters to the file.    
        """
        self.log_writer.log(self.file, 'Entered the elbow_plot method of the KMeansClustering class')
        wcss=[] # Initializing an empty list to save Within Cluster Sum Squares for number of clusters formed
        try:
            for i in range (1,11):
                    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) # initializing the KMeans object
                    kmeans.fit(data) # fitting the data to the KMeans Algorithm
                    wcss.append(kmeans.inertia_)

            plt.plot(range(1,11),wcss) # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('K-means_clustering_Elbow_plot/K-Means_Elbow.PNG') # saving the elbow plot locally
            self.log_writer.log(self.file, 'The K-means_clustering Elbow plot has been saved.')

            # Finding the value of the optimum cluster using the KneeLocator 
            kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.log_writer.log(self.file, 'The optimum number of clusters is: '+str(kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
    
        except Exception as e:
            self.log_writer.log(self.file, 'Exception occured in elbow_plot methods. Exception message:  ' + str(e))
            self.log_writer.log(self.file, 'Finding the number of clusters failed. Exited the elbow_plot method.')
            raise e
        
        return kn.knee
    
## Test:
# try:
#     logging.info('Start Clustering process....')
#     df = pd.read_csv("data_sets/processed_df.csv")
#     X = df.drop(['Class'], axis=1)
#     y = df['Class']
#     logging.info('The X features and y labels have been read Successfully')
# except Exception as e:
#     logging.error(f"An error occurred while reading the data: {str(e)}")
# elbow_plot(X)


# To create clusters in the data_set

    def create_clusters(self, data, k_clusters):
        """ Method Name: create_clusters
            Description: Create a new dataframe consisting of the 'Cluster' Column
        """
        self.log_writer.log(self.file, 'Entered the create_clusters method of the KMeansClustering class')

        try:
            kmeans = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42)
            kmeans_clusters=kmeans.fit_predict(data) #  divide data into clusters

            with open('src/Saved_models/KMeans.pickle', 'wb') as file:
                pickle.dump(kmeans, file)     # saving the KMeans model to directory
            self.log_writer.log(self.file, 'Encoder fitted on data is saved in  Saved_models file.')
            # passing 'Model' as the functions need three parameters

            data['Cluster']=kmeans_clusters  # create a new column in dataset for storing the cluster information
            self.log_writer.log(self.file, 'succesfully created '+str(k_clusters)+ ' clusters. Exited the create_clusters method.')
        
        except Exception as e:
            self.log_writer.log(self.file, 'Exception occured in create_clusters method. Exception message:  ' + str(e))
            self.log_writer.log(self.file, 'Fitting the data to clusters failed. Exited the create_clusters method.')
            raise e
        
        return data

    
