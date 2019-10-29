from flask import Flask,render_template,request
import numpy as np
import pickle
import pandas as pd


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

            yearsExperience = int(request.form['Experience'])
            milesFromMetropolis = int(request.form['Miles'])
            temp['yearsExperience'] = [yearsExperience]
            temp["milesFromMetropolis"] = [milesFromMetropolis]

            job_val = str(request.form['firstList'])
            jobtype = ["jobType_CEO","jobType_CFO", "jobType_CTO", "jobType_JANITOR", "jobType_JUNIOR", "jobType_MANAGER", "jobType_SENIOR", "jobType_VICE_PRESIDENT"]
            for i in range(len(jobtype)):
                if job_val == jobtype[i]:
                    temp[job_val] = 1
                else:
                    temp[jobtype[i]] = 0


            d_val = str(request.form['thirdList'])
            degree = ["degree_BACHELORS", "degree_DOCTORAL", "degree_HIGH_SCHOOL", "degree_MASTERS","degree_NONE"]
            for i in range(len(degree)):
                if d_val == degree[i]:
                    temp[d_val] = 1
                else:
                    temp[degree[i]] = 0


            maj_val = str(request.form['secondList'])
            major = ["major_BIOLOGY", "major_BUSINESS","major_CHEMISTRY", "major_COMPSCI", "major_ENGINEERING", "major_LITERATURE","major_MATH","major_NONE","major_PHYSICS"]
            for i in range(len(major)):
                if maj_val == major[i]:
                    temp[maj_val] = 1
                else:
                    temp[major[i]] = 0


            ind_val = str(request.form['fourthList'])
            industry = ["industry_AUTO","industry_EDUCATION","industry_FINANCE","industry_HEALTH","industry_OIL","industry_SERVICE","industry_WEB"]
            for i in range(len(industry)):
                if ind_val == industry[i]:
                    temp[ind_val] = 1
                else:
                    temp[industry[i]] = 0



            tmp = pd.DataFrame(temp)
            prediction = model.predict(tmp)
            # print([job_val,maj_val,d_val,ind_val,yearsExperience,milesFromMetropolis])
        except ValueError:
            raise ValueError("Input Correct Values")
    return render_template("predict.html",prediction=prediction[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0")