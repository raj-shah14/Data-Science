from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("salarymodel.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=['GET', 'POST'])
def predict():
    temp = {}
    if request.method == "POST":
        try:
            job_val = request.form['firstList']
            # jobtype = ["jobType_CEO","jobType_CFO", "jobType_CTO", "jobType_JANITOR", "jobType_JUNIOR", "jobType_MANAGER", "jobType_SENIOR", "jobType_VICE_PRESIDENT"]
            # joblist = [0] * len(jobtype)            
            maj_val = request.form['secondList']
            d_val = request.form['thirdList']
            ind_val = request.form['fourthList']
            yearsExperience = request.form['Experience']
            milesFromMetropolis = request.form['Miles']
            # print([job_val,maj_val,d_val,ind_val,yearsExperience,milesFromMetropolis])
        except ValueError:
            raise ValueError("Input Correct Values")
    return render_template("predict.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")