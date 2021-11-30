import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


################# 중조립/대조립 공통사항에 대한 전처리를 수행하는 부분 ##########################
path_total = "/content/gdrive/MyDrive/Block_leadtime_machine_learning/Block assembly /"
raw_data = pd.read_csv("raw_data.csv")
raw_data.head()
raw_data.drop(index=[0,1,2],inplace=True)
raw_data.drop("Unnamed: 18",axis=1,inplace=True)
raw_data.drop(["Unnamed: 19","Unnamed: 20","Unnamed: 21"],axis=1,inplace=True)
raw_data.drop("Unnamed: 0",axis=1,inplace=True)
raw_data.rename(columns={"Unnamed: 1": "team", "Unnamed: 2": "D_p", "Unnamed: 3": "S_t", "Unnamed: 4": "P_j", "Unnamed: 5": "B_l",  "Unnamed: 6": "Ass'y", "Unnamed: 7": "PCG", "Unnamed: 8": "PCG_n", "Unnamed: 9": "Com_n", "Unnamed: 10": "Weight", "Unnamed: 11": "L",
                       "Unnamed: 12": "B", "Unnamed: 13": "H", "Unnamed: 14": "PL_LT", "Unnamed: 17": "LT" },inplace=True)
raw_data.drop(["Unnamed: 15","Unnamed: 16"],axis=1,inplace=True)
raw_data.drop(index=[3],inplace=True)

################# 중조립 데이터에 대한 전처리 수행하는 부분 ####################################
is_sub = raw_data['D_p'] == '중조립'
sub_data = raw_data[is_sub]
sub_data = sub_data.drop(['D_p'], axis = 1)    # 불필요 feature(D_p) 삭제
sub_data.drop(["Weight"],axis=1,inplace=True)  # 불필요 feature(Weight) 삭제
sub_data.drop(["Ass'y"],axis=1,inplace=True)   # 불필요 feature(Ass'y) 삭제
sub_data["LT"] =sub_data["LT"].astype(int)
median = sub_data["PL_LT"].median()
sub_data["PL_LT"].fillna(median, inplace=True) # PL_LT feature 중 nan 값을 median 값으로 변경
median = sub_data["H"].median()
sub_data["H"].fillna(median, inplace=True)       # H feature 중 nan 값을 median 값으로 변경
sub_data["L"] =sub_data["L"].astype(float)       # feature의 데이터 타입을 float로 변경
sub_data["B"] =sub_data["B"].astype(float)       # feature의 데이터 타입을 float로 변경
sub_data["H"] =sub_data["H"].astype(float)       # feature의 데이터 타입을 float로 변경
sub_data["PL_LT"] =sub_data["PL_LT"].astype(int) # feature의 데이터 타입을 int로 변경
sub_data["team"] =sub_data["team"].astype(str)   # feature의 데이터 타입을 str로 변경
sub_data["S_t"] =sub_data["S_t"].astype(str)     # feature의 데이터 타입을 str로 변경
sub_data["P_j"] =sub_data["P_j"].astype(str)     # feature의 데이터 타입을 str로 변경
sub_data["B_l"] =sub_data["B_l"].astype(str)     # feature의 데이터 타입을 str로 변경
sub_data["PCG"] =sub_data["PCG"].astype(str)     # feature의 데이터 타입을 str로 변경
sub_data["PCG_n"] =sub_data["PCG_n"].astype(str) # feature의 데이터 타입을 str로 변경
sub_data["Com_n"] =sub_data["Com_n"].astype(str) # feature의 데이터 타입을 str로 변경
sub_data.rename(columns={"D_p": "DP","S_t": "St", "B_l": "BL", "Com_n": "Comn"},inplace=True)
sub_data = sub_data[(sub_data['LT'] >= 5) & (sub_data["LT"] <=11)]
sub_data = sub_data.reset_index()
sub_data.drop(["index"], axis = 1, inplace = True)
BL_LT = sub_data.drop("LT",axis=1)
BL_LT_labels = sub_data["LT"].copy()
list_1=[]
for i in range(len(BL_LT.columns)):
    if not np.issubdtype((BL_LT[(BL_LT.columns[i])]).dtypes, np.int64) and not np.issubdtype((BL_LT[(BL_LT.columns[i])]).dtypes, np.float64) :
        list_1.append(BL_LT.columns[i])
BL_LT_num = BL_LT.drop(list_1,axis=1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),  # 수치형 데이터에 대한 정규화 수행
    ])
from sklearn.compose import ColumnTransformer
num_attribs = list(BL_LT_num)              # 수치형 데이터 분리
cat_attribs = list_1                       # 범주형 데이터 분리
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(sparse=False), cat_attribs), # 범주형 데이터 one-hot-encoding 수행
    ])

BL_LT_prepared = full_pipeline.fit_transform(BL_LT) # 전처리 수행

BL_LT_prepared_train, \
BL_LT_prepared_test, \
BL_LT_labels_train, \
BL_LT_labels_test = train_test_split(
    BL_LT_prepared, BL_LT_labels, test_size = 0.10, random_state = 42) # 훈련:테스트 = 9:1 비율로 분리
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR


ada_et_tree_reg = AdaBoostRegressor(
    ExtraTreeRegressor(max_depth=200, random_state=42), n_estimators=60,
   learning_rate=0.5, random_state=42)
ada_et_tree_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)   # AdaBoostRegressor(ExtraTreeRegressor) 알고리즘 학습 수행
BL_LT_predicted = ada_et_tree_reg.predict(BL_LT_prepared_test) # AdaBoostRegressor(ExtraTreeRegressor) 알고리즘 테스트 수행
ada_et_tree_mape_sub = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0)) # AdaBoostRegressor(ExtraTreeRegressor) 알고리즘 MAPE 산출

ada_svr_reg = AdaBoostRegressor(
    SVR(C=1000,degree=3,kernel='rbf'), n_estimators=30,
   learning_rate=0.1, random_state=42)
ada_svr_reg.fit(BL_LT_prepared_train, BL_LT_labels_train) # AdaBoostRegressor(SVR) 알고리즘 학습 수행
BL_LT_predicted= ada_svr_reg.predict(BL_LT_prepared_test) # AdaBoostRegressor(SVR) 알고리즘 테스트 수행
ada_svr_mape_sub = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0)) # AdaBoostRegressor(SVR) 알고리즘 MAPE 산출


from sklearn.svm import SVR
svm_rbf_reg = SVR(C=10, cache_size=200, coef0=0.0, degree=3,
                           epsilon=0.1, gamma='scale', kernel='rbf',
                           max_iter=-1, shrinking=True, tol=0.001,
                           verbose=False)
svm_rbf_reg.fit(BL_LT_prepared_train,BL_LT_labels_train) # SVR 알고리즘 학습 수행
BL_LT_predicted = svm_rbf_reg.predict(BL_LT_prepared_test) # SVR 알고리즘 테스트 수행
svm_rbf_mape_sub = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0)) # SVR 알고리즘 MAPE 산출



from sklearn.tree import DecisionTreeRegressor
ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=15),n_estimators=200,learning_rate=0.1,loss="exponential",random_state=42)
ada_reg.fit(BL_LT_prepared_train,BL_LT_labels_train) # AdaBoostRegressor(DecisionTreeRegressor) 알고리즘 학습 수행
BL_LT_predicted = ada_reg.predict(BL_LT_prepared_test) # AdaBoostRegressor(DecisionTreeRegressor) 알고리즘 테스트 수행
ada_reg_mape_sub = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0)) # AdaBoostRegressor(DecisionTreeRegressor) 알고리즘 MAPE 산출


print("============중조결과============")
print('AdaBoostRegressor(ExtraTreeRegressor)', ' MAPE :', np.round(ada_et_tree_mape_sub, 4))
print('AdaBoostRegressor(SVR)', ' MAPE :', np.round(ada_svr_mape_sub, 4))
print('SVR', ' MAPE :', np.round(svm_rbf_mape_sub, 4))
print('AdaBoostRegressor(DecisionTreeRegressor)', ' MAPE :', np.round(ada_reg_mape_sub, 4))


################# 중조립 데이터에 대한 전처리 수행하는 부분 ####################################
is_grand = raw_data['D_p'] == '대조립'
grand_data = raw_data[is_grand]
grand_data = grand_data.drop(['D_p'], axis = 1)    # 불필요 feature(D_p) 삭제
grand_data.drop(["Ass'y"],axis=1,inplace=True)     # 불필요 feature(Ass'y) 삭제
median = grand_data["PL_LT"].median()
grand_data["PL_LT"].fillna(median, inplace = True) # PL_LT중 nan 값은 median 값으로 변경
grand_data["L"] =grand_data["L"].astype(float)       # feature의 데이터 타입을 float로 변경
grand_data["B"] =grand_data["B"].astype(float)       # feature의 데이터 타입을 float로 변경
grand_data["H"] =grand_data["H"].astype(float)       # feature의 데이터 타입을 float로 변경
grand_data["PL_LT"] =grand_data["PL_LT"].astype(int) # feature의 데이터 타입을 float로 변경
grand_data["LT"] = grand_data["LT"].astype(int)      # feature의 데이터 타입을 int로 변경
grand_data["team"] =grand_data["team"].astype(str)   # feature의 데이터 타입을 str로 변경
grand_data["S_t"] =grand_data["S_t"].astype(str)     # feature의 데이터 타입을 str로 변경
grand_data["P_j"] =grand_data["P_j"].astype(str)     # feature의 데이터 타입을 str로 변경
grand_data["B_l"] =grand_data["B_l"].astype(str)     # feature의 데이터 타입을 str로 변경
grand_data["PCG"] =grand_data["PCG"].astype(str)     # feature의 데이터 타입을 str로 변경
grand_data["PCG_n"] =grand_data["PCG_n"].astype(str) # feature의 데이터 타입을 str로 변경
grand_data["Com_n"] =grand_data["Com_n"].astype(str) # feature의 데이터 타입을 str로 변경
grand_data.rename(columns={"D_p": "DP","S_t": "St", "B_l": "BL", "Com_n": "Comn"},inplace=True)
grand_data = grand_data[(grand_data['LT'] >= 8) & (grand_data["LT"] <=16)]
grand_data = grand_data.reset_index()
grand_data.drop(["index"], axis = 1, inplace = True)
BL_LT = grand_data.drop("LT",axis=1)
BL_LT_labels = grand_data["LT"].copy()
list_1=[]
for i in range(len(BL_LT.columns)):
    if not np.issubdtype((BL_LT[(BL_LT.columns[i])]).dtypes, np.int64) and not np.issubdtype((BL_LT[(BL_LT.columns[i])]).dtypes, np.float64) :
        list_1.append(BL_LT.columns[i])
BL_LT_num = BL_LT.drop(list_1,axis=1)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),  # 수치형 데이터에 대한 정규화 수행
    ])
from sklearn.compose import ColumnTransformer

num_attribs = list(BL_LT_num)              # 수치형 데이터
cat_attribs = list_1                       # 범주형 데이터

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(sparse=False), cat_attribs), # 범주형 데이터 one-hot-encoding 수행
    ])

BL_LT_prepared = full_pipeline.fit_transform(BL_LT) # 전처리 수행

BL_LT_prepared_train, \
BL_LT_prepared_test, \
BL_LT_labels_train, \
BL_LT_labels_test = train_test_split(BL_LT_prepared, BL_LT_labels, test_size = 0.10, random_state = 42) # 훈련:테스트 = 9:1 비율로 분리


ada_et_tree_reg = AdaBoostRegressor(
    ExtraTreeRegressor(max_depth=200, random_state=42), n_estimators=60,
   learning_rate=0.5, random_state=42)
ada_et_tree_reg.fit(BL_LT_prepared_train,BL_LT_labels_train) # AdaBoostRegressor(ExtraTreeRegressor) 알고리즘 학습 수행
BL_LT_predicted = ada_et_tree_reg.predict(BL_LT_prepared_test) # AdaBoostRegressor(ExtraTreeRegressor) 알고리즘 테스트 수행
ada_et_tree_mape_grand = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0)) # AdaBoostRegressor(ExtraTreeRegressor) 알고리즘 MAPE 산출


ada_svr_reg = AdaBoostRegressor(
    SVR(C=1000,degree=3,kernel='rbf'), n_estimators=30,
   learning_rate=0.1, random_state=42)
ada_svr_reg.fit(BL_LT_prepared_train, BL_LT_labels_train) # AdaBoostRegressor(SVR) 알고리즘 학습 수행
BL_LT_predicted= ada_svr_reg.predict(BL_LT_prepared_test) # AdaBoostRegressor(SVR) 알고리즘 테스트 수행
ada_svr_mape_grand = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0)) # AdaBoostRegressor(SVR) 알고리즘 MAPE 산출


from sklearn.svm import SVR
svm_rbf_reg = SVR(C=10, cache_size=200, coef0=0.0, degree=3,
                           epsilon=0.1, gamma='scale', kernel='rbf',
                           max_iter=-1, shrinking=True, tol=0.001,
                           verbose=False)
svm_rbf_reg.fit(BL_LT_prepared_train,BL_LT_labels_train) # SVR 알고리즘 학습 수행
BL_LT_predicted = svm_rbf_reg.predict(BL_LT_prepared_test) # SVR 알고리즘 테스트 수행
svm_rbf_mape_grand = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0)) # SVR 알고리즘 MAPE 산출


from sklearn.tree import DecisionTreeRegressor
ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=15),n_estimators=200,learning_rate=0.1,loss="exponential",random_state=42)
ada_reg.fit(BL_LT_prepared_train,BL_LT_labels_train) # AdaBoostRegressor(DecisionTreeRegressor) 알고리즘 학습 수행
BL_LT_predicted = ada_reg.predict(BL_LT_prepared_test) # AdaBoostRegressor(DecisionTreeRegressor) 알고리즘 테스트 수행
ada_reg_mape_grand = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0)) # AdaBoostRegressor(DecisionTreeRegressor) 알고리즘 MAPE 산출



print("============대조결과============")
print('AdaBoostRegressor(ExtraTreeRegressor)', ' MAPE :', np.round(ada_et_tree_mape_grand,4))
print('AdaBoostRegressor(SVR)', ' MAPE :', np.round(ada_svr_mape_grand, 4))
print('SVR', ' MAPE :', np.round(svm_rbf_mape_grand, 4))
print('AdaBoostRegressor(DecisionTreeRegressor)', ' MAPE :', np.round(ada_reg_mape_grand, 4))


