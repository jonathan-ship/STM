import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#csv 파일 읽기
ri_MakingLT = pd.read_csv("p_makingdata.csv",encoding="euc-kr")
print(ri_MakingLT.head())
ri_MakingLT.drop(["Unnamed: 0"],axis=1,inplace = True)
print(ri_MakingLT.head())
print(ri_MakingLT.isnull())

list_0 = ["BLOCK","Distribution","NO_Base","NO_SPOOL","NO_Serial","NO_SetPLT","No_row","PaintingLT","Pass","Problem","plan_MakingLT","plan_PaintingLT"]
ri_MakingLT.drop(list_0,axis=1,inplace = True)

#label 만들기
ri_MakingLT_i = ri_MakingLT.drop("MakingLT",axis=1)
ri_MakingLT_labels = ri_MakingLT["MakingLT"].copy()
print(ri_MakingLT_labels)

#Emergency의 dataframe type 바뀌기
ri_MakingLT_i["Emergency"] = ri_MakingLT_i["Emergency"].astype(np.object)

#법주형 데이터의 dataframe을 list_1에 담기
list_1=[]
for i in range(len(ri_MakingLT_i.columns)):
    if not np.issubdtype((ri_MakingLT_i[(ri_MakingLT_i.columns[i])]).dtypes, np.int64) and not np.issubdtype((ri_MakingLT_i[(ri_MakingLT_i.columns[i])]).dtypes, np.float64) :
        list_1.append(ri_MakingLT_i.columns[i])
print(list_1)

#수치형 데이터 dataframe 만들기
ri_MakingLT_num = ri_MakingLT_i.drop(list_1,axis=1)
print(ri_MakingLT_num)

#표준화 처리
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
num_attribs = list(ri_MakingLT_num)             #수치형 데이터
cat_attribs = list_1                     #범주형 데이터
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(sparse=False), cat_attribs),
    ])
ri_MakingLT_prepared = full_pipeline.fit_transform(ri_MakingLT_i)
print(ri_MakingLT_prepared)

# Training Data, Test Data 분리
ri_MakingLT_prepared_train, ri_MakingLT_prepared_test, ri_MakingLT_labels_train, ri_MakingLT_labels_test = train_test_split(ri_MakingLT_prepared, ri_MakingLT_labels, test_size = 0.20, random_state = 42)

# Training Data는 Training Data_really,Training Data_val  분리
ri_MakingLT_prepared_train_re, ri_MakingLT_prepared_train_val, ri_MakingLT_labels_train_re, ri_MakingLT_labels_train_val = train_test_split(ri_MakingLT_prepared_train, ri_MakingLT_labels_train, test_size = 0.25, random_state = 42)

###**ElasticNet**###

# ElasticNet 모델 훈련 시킴
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.001,l1_ratio=0.9,max_iter=100)
elastic_net.fit(ri_MakingLT_prepared_train, ri_MakingLT_labels_train)
ri_MakingLT_predicted = elastic_net.predict(ri_MakingLT_prepared_test)

from sklearn.metrics import mean_squared_error
elastic_net_mse = mean_squared_error(ri_MakingLT_labels_test, ri_MakingLT_predicted)
elastic_net_rmse = np.sqrt(elastic_net_mse)
print(elastic_net_rmse)

from sklearn.metrics import mean_absolute_error
elastic_net_mae = mean_absolute_error(ri_MakingLT_labels_test, ri_MakingLT_predicted)
print(elastic_net_mae)

elastic_net_mape = (np.abs((ri_MakingLT_predicted - ri_MakingLT_labels_test) / ri_MakingLT_labels_test).mean(axis=0))
print(elastic_net_mape)


from sklearn.metrics import mean_squared_log_error
elastic_net_rmsle = np.sqrt(mean_squared_log_error(ri_MakingLT_labels_test, ri_MakingLT_predicted))
print(elastic_net_rmsle)