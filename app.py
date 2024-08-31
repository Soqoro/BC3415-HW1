from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("dbs_linear_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = "Waiting..."
    if request.method == "POST":
        try:
            # Retrieve input from the form
            num = float(request.form.get("rates", 0))
            
            # Make prediction using the loaded model
            prediction = model.predict([[num]])[0][0]
            
            # Round the prediction to 2 decimal places
            result = round(prediction, 2)
        
        except ValueError:
            result = "Invalid input"
        except Exception as e:
            result = f"An error occurred: {e}"

    # Render the template with the result
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
