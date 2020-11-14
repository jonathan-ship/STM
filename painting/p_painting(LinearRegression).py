import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#csv 파일 읽기
ri_PaintingLT = pd.read_csv("p_paintingdata.csv",encoding="euc-kr")
print(ri_PaintingLT .head())

ri_PaintingLT.drop(["Unnamed: 0"],axis=1,inplace = True)
list_0 = ["BLOCK","Distribution","NO_Base","NO_Serial","NO_SetPLT","No_row","Pass","Problem"]
ri_PaintingLT.drop(list_0,axis=1,inplace = True)

#label 만들기
ri_PaintingLT_i = ri_PaintingLT.drop("PaintingLT",axis=1)
ri_PaintingLT_labels = ri_PaintingLT["PaintingLT"].copy()

#Emergency의 dataframe type 바뀌기
ri_PaintingLT_i["Emergency"] = ri_PaintingLT_i["Emergency"].astype(np.object)


#법주형 데이터의 dataframe을 list_1에 담기
list_1=[]
for i in range(len(ri_PaintingLT_i.columns)):
    if not np.issubdtype((ri_PaintingLT_i[(ri_PaintingLT_i.columns[i])]).dtypes, np.int64) and not np.issubdtype((ri_PaintingLT_i[(ri_PaintingLT_i.columns[i])]).dtypes, np.float64) :
        list_1.append(ri_PaintingLT_i.columns[i])
ri_PaintingLT_num = ri_PaintingLT_i.drop(list_1,axis=1)
print(list_1)

#수치형 데이터 dataframe 만들기
ri_PaintingLT_num = ri_PaintingLT_i.drop(list_1,axis=1)
print(ri_PaintingLT_num)

#표준화 처리
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
num_attribs = list(ri_PaintingLT_num)             #수치형 데이터
cat_attribs = list_1                     #범주형 데이터

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(sparse=False), cat_attribs),
    ])
ri_PaintingLT_prepared = full_pipeline.fit_transform(ri_PaintingLT_i)

print(ri_PaintingLT_prepared)
print(ri_PaintingLT_prepared.shape)

# Training Data, Test Data 분리
ri_PaintingLT_prepared_train, ri_PaintingLT_prepared_test, ri_PaintingLT_labels_train, ri_PaintingLT_labels_test = train_test_split(ri_PaintingLT_prepared, ri_PaintingLT_labels, test_size = 0.20, random_state = 42)

# Training Data는 Training Data_really,Training Data_val  분리
ri_PaintingLT_prepared_train_re, ri_PaintingLT_prepared_train_val, ri_PaintingLT_labels_train_re, ri_PaintingLT_labels_train_val = train_test_split(ri_PaintingLT_prepared_train, ri_PaintingLT_labels_train, test_size = 0.25, random_state = 42)


###LinearRegression###

# LinearRegression 모델 훈련 시킴
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=True)
lin_reg.fit(ri_PaintingLT_prepared_train, ri_PaintingLT_labels_train)
ri_PaintingLT_predicted = lin_reg.predict(ri_PaintingLT_prepared_test)

from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(ri_PaintingLT_labels_test, ri_PaintingLT_predicted)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(ri_PaintingLT_labels_test, ri_PaintingLT_predicted)
print(lin_mae)

lin_mape = (np.abs((ri_PaintingLT_predicted - ri_PaintingLT_labels_test) / ri_PaintingLT_labels_test).mean(axis=0))
print(lin_mape)

from sklearn.metrics import mean_squared_log_error
lin_rmsle = np.sqrt(mean_squared_log_error(ri_PaintingLT_labels_test, ri_PaintingLT_predicted))
print(lin_rmsle)
