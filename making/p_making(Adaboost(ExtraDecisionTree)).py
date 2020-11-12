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

###**Adaboost(ExtraDecisionTree)**###

# **Adaboost(ExtraDecisionTree)** 모델 훈련 시킴
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import ExtraTreeRegressor
ada_et_tree_reg = AdaBoostRegressor(
    ExtraTreeRegressor(max_depth=11, random_state=42), n_estimators=200,
   learning_rate=0.5, random_state=42)
ada_et_tree_reg.fit(ri_MakingLT_prepared_train_re,ri_MakingLT_labels_train_re)
ri_MakingLT_predicted = ada_et_tree_reg.predict(ri_MakingLT_prepared_train_val)
print("Predictions:", ri_MakingLT_predicted)
print("Labels:", list(ri_MakingLT_labels_train_val))
#rmse측정
from sklearn.metrics import mean_squared_error
ada_et_tree_reg_mse_val = mean_squared_error(ri_MakingLT_labels_train_val, ri_MakingLT_predicted)
print(ada_et_tree_reg_mse_val)


#훈렬한 모틸이 overfitting인지 unoverfitting인지 판단(방법1:학습곡선)
def plot_learning_curves_sup(model,X_train,X_val,y_train,y_val):
  train_errors, val_errors = [], []
  for m in range(1,len(X_train),3000):
    model.fit(X_train[:m], y_train[:m])
    y_train_predict = model.predict(X_train[:m])
    y_val_predict = model.predict(X_val)
    train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
    val_errors.append(mean_squared_error(y_val_predict,y_val))
  plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
  plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
  plt.legend(loc="upper right", fontsize=14)
  plt.xlabel("Training set size/3000", fontsize=14)
  plt.ylabel("RMSE", fontsize=14)

plot_learning_curves_sup(ada_et_tree_reg, ri_MakingLT_prepared_train_re, ri_MakingLT_prepared_train_val, ri_MakingLT_labels_train_re, ri_MakingLT_labels_train_val)
plt.axis([0, len(ri_MakingLT_prepared_train_re)/3000, 0, 20])
plt.show()

#훈렬한 모틸이 overfitting인지 unoverfitting인지 판단(방법2:교차검증)
from sklearn.model_selection import cross_val_score
ada_reg_scores = cross_val_score(ada_et_tree_reg, ri_MakingLT_prepared_train,ri_MakingLT_labels_train,scoring="neg_mean_squared_error", cv=10)
ada_reg_rmse_scores = np.sqrt(-ada_reg_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

print(display_scores(ada_reg_rmse_scores))


#####RandomizedSearch 방법을 이용해 최적 paramater 찾기
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {
        'n_estimators': randint(low=1, high=500),
        'loss':["linear","square","exponential"],
        'learning_rate': [0.1,0.5,1],
    }
ada_et_tree_reg = AdaBoostRegressor(
    ExtraTreeRegressor(max_depth=11, random_state=42),random_state=42)
ada_et_tree_reg_search = RandomizedSearchCV(ada_et_tree_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
grid_result = ada_et_tree_reg_search.fit(ri_MakingLT_prepared_train,ri_MakingLT_labels_train)
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

#마지막으로 최적 paramater를 넣고 test세트에 대하여 장확도를 예측하기
ada_et_tree_reg = AdaBoostRegressor(base_estimator=ExtraTreeRegressor(ccp_alpha=0.0,
                                                    criterion='mse',
                                                    max_depth=11,
                                                    max_features='auto',
                                                    max_leaf_nodes=None,
                                                    min_impurity_decrease=0.0,
                                                    min_impurity_split=None,
                                                    min_samples_leaf=1,
                                                    min_samples_split=2,
                                                    min_weight_fraction_leaf=0.0,
                                                    random_state=42,
                                                    splitter='random'),
                  learning_rate=0.1, loss='exponential', n_estimators=150,
                  random_state=42)
ada_et_tree_reg.fit(ri_MakingLT_prepared_train, ri_MakingLT_labels_train)
ri_MakingLT_predicted = ada_et_tree_reg.predict(ri_MakingLT_prepared_test)

from sklearn.metrics import mean_squared_error
ada_et_tree_reg_mse = mean_squared_error(ri_MakingLT_labels_test, ri_MakingLT_predicted)
ada_et_tree_reg_rmse = np.sqrt(ada_et_tree_reg_mse)
print(ada_et_tree_reg_rmse)

from sklearn.metrics import mean_absolute_error
ada_et_tree_reg_mae = mean_absolute_error(ri_MakingLT_labels_test, ri_MakingLT_predicted)
print(ada_et_tree_reg_mae)

ada_et_tree_reg_mape = (np.abs((ri_MakingLT_predicted - ri_MakingLT_labels_test) / ri_MakingLT_labels_test).mean(axis=0))
print(ada_et_tree_reg_mape)


from sklearn.metrics import mean_squared_log_error
ada_et_tree_reg_rmsle = np.sqrt(mean_squared_log_error(ri_MakingLT_labels_test, ri_MakingLT_predicted))
print(ada_et_tree_reg_rmsle)