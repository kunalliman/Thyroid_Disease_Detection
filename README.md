# Thyroid Level Detection

A solution for transforming the way medical teams detect thyroid diseases, leading to faster and more accurate diagnoses.

## Problem Statement

The goal of this project is to create a classification model using machine learning algorithms to predict the type of thyroid disease a person has based on the patient's data. This solution aids medical professionals in achieving:

- Accurate predictions of thyroid diseases.
- Efficiency in detecting the type of thyroid disease.
- Reducing human error in the diagnostic process.

## Project Flow

### Model Training and Saving

1. **Data Ingestion (Validation):** Ensuring that the data used for model training is valid and accurate.
2. **Data Preprocessing:** Preparing and cleaning the data for training.
3. **Data Clustering:** Grouping data for efficient model training.
4. **Train-Test Split for Each Cluster:** Splitting the data into training and testing datasets for each cluster.
5. **Training and Model Evaluation:** Training the classification model for each cluster and evaluating its performance.
6. **Save Model:** Saving the trained models for future use.
7. **Prediction:** Making predictions using the saved models.

All the training code is located in the src directory and can be executed by running the __main__.py file.
[My Document](src\__main__.py)

## Data Description and EDA

For a detailed data description and exploratory data analysis (EDA), please refer to the PowerPoint (PPT) and Jupyter Notebook files available in the "notebook_files_&_PPT" directory.
[My Document](notebook_files_&_PPT)


### Predicting on Live Data with a User Interface (UI)

1. **Uploading File:** User uploads a file containing patient data.
2. **Data Ingestion (Validation):** Ensuring that the uploaded data is valid and accurate.
3. **Data Preprocessing:** Preparing and cleaning the uploaded data.
4. **Data Clustering:** Grouping the data for efficient prediction.
5. **Predicting by Saved Models for Each Cluster:** Using the saved models to make predictions.
6. **Return the Uploaded File with Prediction Results to Download:** Providing the user with the option to download the file with prediction results.

All the code can be tested using the test code written at the bottom of Pred_valid_insert.py or by running app.py for accessing it via a web page.
[My Document](Pred_valid_insert.py)    For app: [My Document](app.py) 

[![Video](https://github.com/kunalliman/Thyroid_Level_Prediction/blob/main/notebook_files_%26_PPT/Web_img.png)](["notebook_files_&_PPT\demo_video.mp4](https://github.com/kunalliman/Thyroid_Level_Prediction/blob/main/notebook_files_%26_PPT/Demo_video.mp4))

## Conclusion

The thyroid disease detection solution empowers medical teams to detect thyroidal diseases in patients based on features, symptoms, and the quantity of 4 hormones. This solution eliminates human error, increases efficiency in diagnosing thyroid diseases, and ensures that necessary medical actions are taken by the concerned authorities.

## Tools Used

This project utilizes the following tools and technologies:

- Python
- Flask
- HTML



