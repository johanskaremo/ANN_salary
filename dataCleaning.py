import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import sys
pd.set_option('display.max_columns', None)



# def cleanData(filename):
#     df = pd.read_csv(filename)
#     indexEmp = df[(df["employment_type"] != "FT")].index
#     df.drop(indexEmp, inplace=True)
#     data = df.drop(columns=["employment_type", "salary", "salary_currency"])
#     cutDf = data.copy()
#     cutDf["salary_in_usd"] = pd.cut(cutDf["salary_in_usd"].values, bins= 5, labels=[0, 1, 2, 3, 4])
#     #classes
#     year = LabelEncoder()
#     exp_lvl = LabelEncoder()
#     job_title = LabelEncoder()
#     country = LabelEncoder()
#     comp_size = LabelEncoder()
#
#     cutDf = cutDf.iloc[:, :].values
#     tail = cutDf[-10:]
#     print(tail)
#
#     #fit encoded variables to specific columns
#     cutDf[:, 0] = year.fit_transform(cutDf[:, 0])
#     cutDf[:, 1] = exp_lvl.fit_transform(cutDf[:, 1])
#     cutDf[:, 2] = job_title.fit_transform(cutDf[:, 2])
#     cutDf[:, 4] = country.fit_transform(cutDf[:, 4])
#     cutDf[:, 5] = comp_size.fit_transform(cutDf[:, 5])
#
#     tail2 = cutDf[-10:]
#     print(tail2)
#
#     return cutDf

def cleanData(data):

    indexEmp = data[(data["employment_type"] != "FT")].index
    data.drop(indexEmp, inplace=True)
    data = data.drop(columns=["employment_type" , "salary", "salary_currency"])
    cutData = data.copy()
    cutData["salary_in_usd"] = pd.cut(cutData["salary_in_usd"].values, bins=5, labels=[0, 1, 2, 3, 4])
    #classes
    year = LabelEncoder()
    exp_lvl = LabelEncoder()
   # job_title = LabelEncoder()
    #country = LabelEncoder()
    comp_size = LabelEncoder()
    min_max_scaler = MinMaxScaler()
    #cutData = cutData.iloc[:,:].values

    #fit encoded variables to specific columns
    cutData.iloc[:, 0] = year.fit_transform(cutData.iloc[:, 0])
    cutData.iloc[:, 1] = exp_lvl.fit_transform(cutData.iloc[:, 1])
    cutData.iloc[:, 5] = comp_size.fit_transform(cutData.iloc[:, 5])

    cutData[['work_year', 'experience_level', 'company_size']] = min_max_scaler.fit_transform(cutData[['work_year', 'experience_level', 'company_size']])
    #year = 0, experiance level = 1, job title = 2, country = 4, company size = 5
    #salary = 3

    freq_job = cutData.groupby("job_title").size() / len(cutData)
    cutData["job_title"] = cutData.job_title.map(freq_job)
    freq_loc = cutData.groupby("company_location").size() / len(cutData)
    cutData["company_location"] = cutData.company_location.map(freq_loc)
    return cutData