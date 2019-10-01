import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
import xgboost as xgb



model = pickle.load(open("salarymodel.pkl","rb"))

def flatten(arr,res):
    for i in arr:
        if type(i) == list:
            flatten(i,res)
        else:
            res.append(i)
    return res

temp = {}


yearsExperience = 20
milesFromMetropolis = 50

temp['yearsExperience'] = [yearsExperience]
temp["milesFromMetropolis"] = [milesFromMetropolis]

jobtype = ["jobType_CEO","jobType_CFO", "jobType_CTO", "jobType_JANITOR", "jobType_JUNIOR", "jobType_MANAGER", "jobType_SENIOR", "jobType_VICE_PRESIDENT"]
job_val = [0]*len(jobtype)
job_val[4] = 1
for i in range(len(jobtype)):
    temp[jobtype[i]] = [job_val[i]]


degree = ["degree_BACHELORS", "degree_DOCTORAL", "degree_HIGH_SCHOOL", "degree_MASTERS","degree_NONE"]
d_val = [0]*len(degree)
d_val[2] = 1
for i in range(len(degree)):
    temp[degree[i]] = [d_val[i]]

major = ["major_BIOLOGY", "major_BUSINESS","major_CHEMISTRY", "major_COMPSCI", "major_ENGINEERING", "major_LITERATURE","major_MATH","major_NONE","major_PHYSICS"]
maj_val = [0]*len(major)
maj_val[5] = 1
for i in range(len(major)):
    temp[major[i]] = [maj_val[i]]

industry = ["industry_AUTO","industry_EDUCATION","industry_FINANCE","industry_HEALTH","industry_OIL","industry_SERVICE","industry_WEB"]
ind_val = [0]*len(industry)
ind_val[1] = 1
for i in range(len(industry)):
    temp[industry[i]] = [ind_val[i]]

tmp = pd.DataFrame(temp)
print(tmp)

print(model.predict(tmp))
order = model.get_booster().feature_names
# print(tmp.columns,order)