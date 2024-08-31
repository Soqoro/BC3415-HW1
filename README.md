# DBS Share Price Prediction App

This project is a web application built with Flask that predicts future DBS share prices based on the SGD to USD exchange rate using a pre-trained linear regression model.

### Setup

-   Follow these steps to set up and run the application locally:

1. Create a Virtual Environment:

-   Before running the application, create a virtual environment to manage dependencies:
    `python -m venv venv`

2. Activate the Virtual Environment:

    - On Windows: `.\venv\Scripts\activate`
    - On Mac/Linux: `source venv/bin/activate`

3. Install Required Packages:
   `pip install -r requirements.txt`

4. Run the Python Model Script: - This script will create the linear regression model and save it for use by the Flask app:
   `python model.py`

5. Run the Flask Application: - Start the Flask web server:
   `python app.py`

6. Access the Application:
    - Open your web browser and navigate to http://127.0.0.1:5000/ to use the app.

### Links

-   [PythonAnywhere Deployment](https://gnoriqaus.pythonanywhere.com/)
-   [Render Deployment](https://bc3415-hw1-p5p2.onrender.com)
