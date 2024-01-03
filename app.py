from flask import Flask, request, render_template, send_file, jsonify
from Pred_valid_insert import pred_data_validation_and_insertion
from Predict_from_model import prediction
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    try:
        uploaded_file = request.files["file"]
        uploaded_file_name = uploaded_file.filename     # Creating file_path ,uploaded_file_name variables
        file_path = f'Pred_Uploaded_File/{uploaded_file_name}'

        uploaded_file.save(f'Pred_Uploaded_File/{uploaded_file_name}')

        pred_valid_insert = pred_data_validation_and_insertion(file_path ,uploaded_file_name) #object initialization
        pred_valid_insert.pred_data_validate_and_insert()   ## This function starts all the steps(functions) to validate and insert data in Pred_Valid_dataset

        pred = prediction(uploaded_file_name) #object initialization
        pred.predictionFromModel() # Will create prediction files for each cluster
        pred.save_pred_results('Pred_Uploaded_File/InputFile.csv')    
# Saves the result from the predicted files to the uploaded file and save it by as 'Predicted_results.csv' in PREDICTIONs
        
        # Create the download link for the client
        download_link = '/download/PREDICTIONs/Predicted_Results.csv'  # Update the path and file name

        # Return the download link as a response
        return jsonify({"DownloadLink": download_link})

    except Exception as e:
        error_message = str(e)
        return f"Error: {error_message}"
    

@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    # Build the full file path based on the filename
    full_path = os.path.join(app.root_path, filename)

    # Return the file as an attachment for download
    return send_file(full_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=5000)   