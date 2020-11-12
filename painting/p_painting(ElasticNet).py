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

###**ElasticNet**###

# ElasticNet 모델 훈련 시킴
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1,l1_ratio=0.2)
elastic_net.fit(ri_PaintingLT_prepared_train_re,ri_PaintingLT_labels_train_re)
ri_PaintingLT_predicted = elastic_net.predict(ri_PaintingLT_prepared_train_val)
print("Predictions:", ri_PaintingLT_predicted)
print("Labels:", list(ri_PaintingLT_labels_train_val))
#rmse측정
from sklearn.metrics import mean_squared_error
elastic_net_mse_val = mean_squared_error(ri_PaintingLT_labels_train_val, ri_PaintingLT_predicted)
print(elastic_net_mse_val)

#훈렬한 모틸이 overfitting인지 unoverfitting인지 판단(방법1:학습곡선)
def plot_learning_curves(model,X_train,X_val,y_train,y_val):
  train_errors, val_errors = [], []
  for m in range(1,len(X_train),50):
    model.fit(X_train[:m], y_train[:m])
    y_train_predict = model.predict(X_train[:m])
    y_val_predict = model.predict(X_val)
    train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
    val_errors.append(mean_squared_error(y_val_predict,y_val))
  plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")    #np.sqrt(x) ： 计算数组各元素的平方根  也就是得出来的是RMESE = （MSE）^（0.5）
  plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
  plt.legend(loc="upper right", fontsize=14)
  plt.xlabel("Training set size/50", fontsize=14)
  plt.ylabel("RMSE", fontsize=14)
  # print(train_errors)
  # print(val_errors)
  # print(np.sqrt(val_errors))

plot_learning_curves(elastic_net, ri_PaintingLT_prepared_train_re, ri_PaintingLT_prepared_train_val, ri_PaintingLT_labels_train_re, ri_PaintingLT_labels_train_val)
plt.axis([0, len(ri_PaintingLT_prepared_train_re)/50, 0, 20])
plt.show()

#훈렬한 모틸이 overfitting인지 unoverfitting인지 판단(방법2:교차검증)
from sklearn.model_selection import cross_val_score
elastic_net_scores = cross_val_score(elastic_net, ri_PaintingLT_prepared_train,ri_PaintingLT_labels_train,scoring="neg_mean_squared_error", cv=10)
elastic_net_rmse_scores = np.sqrt(-elastic_net_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

print(display_scores(elastic_net_rmse_scores))

#####GridSearch 방법을 이용해 최적 paramater 찾기
from sklearn.model_selection import GridSearchCV
elastic_net_reg = ElasticNet()
param_grid = [
  {"max_iter": [1, 5, 10,100,1000],
                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}
 ]
grid = GridSearchCV(estimator=elastic_net_reg, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)
grid_result = grid.fit(ri_PaintingLT_prepared_train,ri_PaintingLT_labels_train)
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

#마지막으로 최적 paramater를 넣고 test세트에 대하여 장확도를 예측하기
elastic_net = ElasticNet(alpha=0.001,l1_ratio=0.9,max_iter=1000)
elastic_net.fit(ri_PaintingLT_prepared_train, ri_PaintingLT_labels_train)
ri_PaintingLT_predicted = elastic_net.predict(ri_PaintingLT_prepared_test)

from sklearn.metrics import mean_squared_error
elastic_net_mse = mean_squared_error(ri_PaintingLT_labels_test, ri_PaintingLT_predicted)
elastic_net_rmse = np.sqrt(elastic_net_mse)
print(elastic_net_rmse)

from sklearn.metrics import mean_absolute_error
elastic_net_mae = mean_absolute_error(ri_PaintingLT_labels_test, ri_PaintingLT_predicted)
print(elastic_net_mae)

elastic_net_mape = (np.abs((ri_PaintingLT_predicted - ri_PaintingLT_labels_test) / ri_PaintingLT_labels_test).mean(axis=0))
print(elastic_net_mape)


from sklearn.metrics import mean_squared_log_error
elastic_net_rmsle = np.sqrt(mean_squared_log_error(ri_PaintingLT_labels_test, ri_PaintingLT_predicted))
print(elastic_net_rmsle)