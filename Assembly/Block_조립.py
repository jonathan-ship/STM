import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt


print("####################################################################################################################")
print(                                       " # Preprocessing start#"                                                  )
print("####################################################################################################################")
raw_data = pd.read_csv("조립_리드타임_실적.csv")
# print(raw_data.head())
raw_data.drop(index=[0,1,2],inplace=True)
raw_data.drop(["Unnamed: 0","Unnamed: 18","Unnamed: 19","Unnamed: 20","Unnamed: 21"],axis=1,inplace=True)
# print(raw_data)
raw_data.rename(columns={"Unnamed: 1": "team", "Unnamed: 2": "D_p", "Unnamed: 3": "S_t", "Unnamed: 4": "P_j", "Unnamed: 5": "B_l",
                         "Unnamed: 6": "Ass'y", "Unnamed: 7": "PCG", "Unnamed: 8": "PCG_n", "Unnamed: 9": "Com_n",
                         "Unnamed: 10": "Weight", "Unnamed: 11": "L", "Unnamed: 12": "B", "Unnamed: 13": "H",
                         "Unnamed: 14": "PL_LT", "Unnamed: 17": "LT" },inplace=True)
raw_data.drop(["Unnamed: 15","Unnamed: 16"],axis=1,inplace=True)
# print(raw_data)
raw_data.drop(index=[3],inplace=True)
raw_data.drop(["Weight"],axis=1,inplace=True)
raw_data.drop(["Ass'y"],axis=1,inplace=True)
raw_data["LT"] =raw_data["LT"].astype(int)
raw_data.dropna(axis=0,how='any',inplace=True) #drop all rows that have any NaN values
raw_data = raw_data.reset_index()
raw_data.drop(["index"],axis=1,inplace=True)
# print(raw_data)
raw_data["L"] =raw_data["L"].astype(float)
raw_data["B"] =raw_data["B"].astype(float)
raw_data["H"] =raw_data["H"].astype(float)
raw_data["PL_LT"] =raw_data["PL_LT"].astype(int)
raw_data["team"] =raw_data["team"].astype(str)
raw_data["D_p"] =raw_data["D_p"].astype(str)
raw_data["S_t"] =raw_data["S_t"].astype(str)
raw_data["P_j"] =raw_data["P_j"].astype(str)
raw_data["B_l"] =raw_data["B_l"].astype(str)
# raw_data["Ass'y"] =raw_data["Ass'y"].astype(str)
raw_data["PCG"] =raw_data["PCG"].astype(str)
raw_data["PCG_n"] =raw_data["PCG_n"].astype(str)
raw_data["Com_n"] =raw_data["Com_n"].astype(str)
raw_data.rename(columns={"D_p": "DP","S_t": "St", "B_l": "BL", "Com_n": "Comn"},inplace=True)
raw_data = raw_data.reset_index()
raw_data.drop(["index"],axis=1,inplace=True)

BL_LT = raw_data.drop("LT",axis=1)
BL_LT_labels = raw_data["LT"].copy()


list_1=[]
for i in range(len(BL_LT.columns)):
    if not np.issubdtype((BL_LT[(BL_LT.columns[i])]).dtypes, np.int64) and not np.issubdtype((BL_LT[(BL_LT.columns[i])]).dtypes, np.float64) :
        list_1.append(BL_LT.columns[i])
# print(list_1)

BL_LT_num = BL_LT.drop(list_1,axis=1)

num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

num_attribs = list(BL_LT_num)
cat_attribs = list_1

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(sparse=False), cat_attribs),
    ])

BL_LT_prepared = full_pipeline.fit_transform(BL_LT)

# Training Data, Test Data 분리
BL_LT_prepared_train, BL_LT_prepared_test, BL_LT_labels_train, BL_LT_labels_test = train_test_split(BL_LT_prepared, BL_LT_labels, test_size = 0.20, random_state = 42)
print()
print()
print()
print("####################################################################################################################")
print(                                       " # Preprocessing finished #"                                                   )
print("####################################################################################################################")
print()
print()
print()
print("####################################################################################################################")
print(                                              "# Training start #"                                                    )
print("####################################################################################################################")

####################################################################################################################
                                              # LinearRegression #
####################################################################################################################
lin_reg = LinearRegression()
lin_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
BL_LT_predicted = lin_reg.predict(BL_LT_prepared_test)


lin_mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

lin_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(lin_mae)

lin_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("LinearRegression: "+str(lin_mape))


lin_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(lin_rmsle)


####################################################################################################################
                                              # Ridge #
####################################################################################################################
ridge_reg = Ridge(alpha=0.1,solver="auto",random_state=42) ##'cholesky'使用标准的scipy.linalg.solve函数来获得闭合形式的解。使用 Cholesky 法进行矩阵分解对公式 4-9 进行变形
ridge_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
BL_LT_predicted = ridge_reg.predict(BL_LT_prepared_test)


ridge_reg_mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
ridge_reg_rmse = np.sqrt(ridge_reg_mse)
# print(ridge_reg_rmse)


ridge_reg_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(ridge_reg_mae)

ridge_reg_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("ridge_reg_mape: "+str(ridge_reg_mape))


ridge_reg_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(ridge_reg_rmsle)





####################################################################################################################
                                              # Lasso #
####################################################################################################################
lasso_reg = Lasso(alpha=0.1,normalize=True)
lasso_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
BL_LT_predicted = lasso_reg.predict(BL_LT_prepared_test)

lasso_reg_mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
lasso_reg_rmse = np.sqrt(lasso_reg_mse)
# print(lasso_reg_rmse)


lasso_reg_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(lasso_reg_mae)

lasso_reg_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("lasso_reg_mape: "+str(lasso_reg_mape))


lasso_reg_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(lasso_reg_rmsle)



####################################################################################################################
                                              # LinearSVR #
####################################################################################################################
svm_reg = LinearSVR(epsilon=0.1, random_state=42)
svm_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
BL_LT_predicted = svm_reg.predict(BL_LT_prepared_test)

SVR__mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
SVR__rmse = np.sqrt(SVR__mse)
# print(SVR__rmse)

SVR_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(SVR_mae)

SVR_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("Svm_Linear: "+str(SVR_mape))

SVR_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(SVR_rmsle)


####################################################################################################################
                                              # SVM_rbf #
####################################################################################################################
svm_rbf_reg = SVR(C=10, epsilon=0.255, kernel='rbf')
svm_rbf_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
BL_LT_predicted = svm_rbf_reg.predict(BL_LT_prepared_test)

svm_rbf_reg_mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
svm_rbf_reg_rmse = np.sqrt(svm_rbf_reg_mse)
# print(svm_rbf_reg_rmse)

svm_rbf_reg_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(svm_rbf_reg_mae)

svm_rbf_reg_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("SVM_rbf: "+str(svm_rbf_reg_mape))

svm_rbf_reg_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(svm_rbf_reg_rmsle)



####################################################################################################################
                                              # RandomForestRegressor #
####################################################################################################################
rnd_reg = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
BL_LT_predicted =  rnd_reg.predict(BL_LT_prepared_test)

RandomForest_mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
RandomForest_rmse = np.sqrt(RandomForest_mse)
# print(RandomForest_rmse)

RandomForest_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(RandomForest_mae)

RandomForest_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("RandomForest: "+str(RandomForest_mape))

RandomForest_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(RandomForest_rmsle)



# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.svm import SVR
# ada_svr_reg = AdaBoostRegressor(
#     SVR(C=20, epsilon=0.155, kernel='rbf'), n_estimators=50,
#    learning_rate=0.1, random_state=42)
# ada_svr_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
# BL_LT_predicted =  ada_svr_reg.predict(BL_LT_prepared_test)
# AdaB_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print(AdaB_mape)


####################################################################################################################
                                              # DecisionTreeRegressor #
####################################################################################################################
tree_reg = DecisionTreeRegressor(max_depth=25)
tree_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
BL_LT_predicted = tree_reg.predict(BL_LT_prepared_test)
tree_mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)

tree_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(tree_mae)

tree_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("decision_tree: "+str(tree_mape))

tree_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(tree_rmsle)

#모델 세부 튜닝(예제)
# from sklearn.model_selection import GridSearchCV
# param_grid = [
#   { "max_depth":list(range(1, 100))}
#  ]
# grid = GridSearchCV(estimator=ExtraTreeRegressor(random_state=42), param_grid=param_grid, verbose=2, cv=10)
# grid_result = grid.fit(BL_LT_prepared_train,BL_LT_labels_train)
# print('Best Score: ', grid_result.best_score_)
# print('Best Params: ', grid_result.best_params_)


####################################################################################################################
                                              # ExtraTreeRegressor #
####################################################################################################################
Et_tree_reg = ExtraTreeRegressor(max_depth=13, random_state=42)
Et_tree_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
BL_LT_predicted = Et_tree_reg.predict(BL_LT_prepared_test)


Et_tree_mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
Et_tree_rmse = np.sqrt(Et_tree_mse)
# print(Et_tree_rmse)

Et_tree_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(Et_tree_mae)

Et_tree_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("Et_tree: "+str(Et_tree_mape))

Et_tree_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(Et_tree_rmsle)



####################################################################################################################
                                              # AdaBoostRegressor(ExtraTreeRegressor) #
####################################################################################################################
ada_et_tree_reg = AdaBoostRegressor(
    ExtraTreeRegressor(max_depth=45, random_state=42), n_estimators=200,
   learning_rate=0.5, random_state=42)
ada_et_tree_reg.fit(BL_LT_prepared_train,BL_LT_labels_train)
BL_LT_predicted = ada_et_tree_reg.predict(BL_LT_prepared_test)


ada_et_tree_mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
ada_et_tree_rmse = np.sqrt(ada_et_tree_mse)
# print(ada_et_tree_rmse)

ada_et_tree_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(ada_et_tree_mae)

ada_et_tree_mape = (np.abs((BL_LT_predicted - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("Ada_et_tree: "+str(ada_et_tree_mape))

ada_et_tree_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(ada_et_tree_rmsle)




####################################################################################################################
                                              # DNN #
####################################################################################################################
dl_m_col = len(BL_LT_prepared_train[0])
deeplearning_model = Sequential()

# Input Layer
deeplearning_model.add(Dense(150, input_dim = dl_m_col, activation='relu'))
deeplearning_model.add(Dropout(0.3))

# Hidden Layer 1
deeplearning_model.add(Dense(150, activation='relu'))
deeplearning_model.add(Dropout(0.3))

# Hidden Layer 2
deeplearning_model.add(Dense(150, activation='relu'))
deeplearning_model.add(Dropout(0.3))

# Hidden Layer 3
deeplearning_model.add(Dense(100, activation='relu'))
deeplearning_model.add(Dropout(0.3))

# Output Layer
deeplearning_model.add(Dense(1))
deeplearning_model.summary()

deeplearning_model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])

dl_m_hist = deeplearning_model.fit(BL_LT_prepared_train,BL_LT_labels_train)

BL_LT_predicted = deeplearning_model.predict(BL_LT_prepared_test)

a = list(np.array(BL_LT_predicted).flatten())


deeplearning_model_mse = mean_squared_error(BL_LT_labels_test, BL_LT_predicted)
deeplearning_model_rmse = np.sqrt(deeplearning_model_mse)
# print(deeplearning_model_rmse)

deeplearning_model_mae = mean_absolute_error(BL_LT_labels_test, BL_LT_predicted)
# print(deeplearning_model_mae)

deeplearning_model_mape = (np.abs((a - BL_LT_labels_test) / BL_LT_labels_test).mean(axis=0))
# print("DNN: "+str(deeplearning_model_mape))

deeplearning_model_rmsle = np.sqrt(mean_squared_log_error(BL_LT_labels_test, BL_LT_predicted))
# print(deeplearning_model_rmsle)

print("####################################################################################################################")
print(                                              "# Training finished #"                                                  )
print("####################################################################################################################")
print()
print()

print("-----------Test_set_features----------------------")
print(BL_LT_prepared_test)
print("Test_set_shape: "+str(BL_LT_prepared_test.shape))
print()
print()
print("-----------Test_set_labels----------------------")
print(BL_LT_labels_test)
print("shape: "+str(BL_LT_labels_test.shape))
print()
print()




print("####################################################################################################################")
print(                                              "# Test results #"                                                  )
print("####################################################################################################################")
print()
print()

print("####################################################################################################################")
print(                                              "# LinearRegression #"                                                  )
print("####################################################################################################################")

print("mae: "+str(lin_mae))
print("mape: "+str(lin_mape*100)+" %")
print("rmse: "+str(lin_rmse))
print("rmsle: "+str(lin_rmsle*100)+" %")
lin_list = [lin_mae,lin_mape*100,lin_rmse,lin_rmsle*100]
print()
print()
print()
print("####################################################################################################################")
print(                                              "# Ridge #"                                                             )
print("####################################################################################################################")

print("mae: "+str(ridge_reg_mae))
print("mape: "+str(ridge_reg_mape*100)+" %")
print("rmse: "+str(ridge_reg_rmse))
print("rmsle: "+str(ridge_reg_rmsle*100)+" %")
Ridge_list = [ridge_reg_mae,ridge_reg_mape*100,ridge_reg_rmse,ridge_reg_rmsle*100]
print()
print()
print()
print("####################################################################################################################")
print(                                              "# Lasso #"                                                             )
print("####################################################################################################################")

print("mae: "+str(lasso_reg_mae))
print("mape: "+str(lasso_reg_mape*100)+" %")
print("rmse: "+str(lasso_reg_rmse))
print("rmsle: "+str(lasso_reg_rmsle*100)+" %")
Lasso_list = [lasso_reg_mae,lasso_reg_mape*100,lasso_reg_rmse,lasso_reg_rmsle*100]
print()
print()
print()
print("####################################################################################################################")
print(                                              "# SVM_Linear #"                                                        )
print("####################################################################################################################")

print("mae: "+str(SVR_mae))
print("mape: "+str(SVR_mape*100)+" %")
print("rmse: "+str(SVR__rmse))
print("rmsle: "+str(SVR_rmsle*100)+" %")
LinearSVR_list = [SVR_mae,SVR_mape*100,SVR__rmse,SVR_rmsle*100]
print()
print()
print()
print("####################################################################################################################")
print(                                              "# SVM_kernel_rbf #"                                                    )
print("####################################################################################################################")

print("mae: "+str(svm_rbf_reg_mae))
print("mape: "+str(svm_rbf_reg_mape*100)+" %")
print("rmse: "+str(svm_rbf_reg_rmse))
print("rmsle: "+str(svm_rbf_reg_rmsle*100)+" %")
SVM_kernel_rbf_list = [svm_rbf_reg_mae,svm_rbf_reg_mape*100,svm_rbf_reg_rmse,svm_rbf_reg_rmsle*100]
print()
print()
print()
print("####################################################################################################################")
print(                                              "# RandomForestRegressor #"                                             )
print("####################################################################################################################")

print("mae: "+str(RandomForest_mae))
print("mape: "+str(RandomForest_mape*100)+" %")
print("rmse: "+str(RandomForest_rmse))
print("rmsle: "+str(RandomForest_rmsle*100)+" %")
RandomForestRegressor_list = [RandomForest_mae,RandomForest_mape*100,RandomForest_rmse,RandomForest_rmsle*100]
print()
print()
print()
print("####################################################################################################################")
print(                                              "# DecisionTreeRegressor #"                                             )
print("####################################################################################################################")

print("mae: "+str(tree_mae))
print("mape: "+str(tree_mape*100)+" %")
print("rmse: "+str(tree_rmse))
print("rmsle: "+str(tree_rmsle*100)+" %")
DecisionTreeRegressor_list = [tree_mae,tree_mape*100,tree_rmse,tree_rmsle*100]
print()
print()
print()
print("####################################################################################################################")
print(                                              "# ExtraTreeRegressor #"                                                )
print("####################################################################################################################")

print("mae: "+str(Et_tree_mae))
print("mape: "+str(Et_tree_mape*100)+" %")
print("rmse: "+str(Et_tree_rmse))
print("rmsle: "+str(Et_tree_rmsle*100)+" %")
ExtraTreeRegressor_list = [Et_tree_mae,Et_tree_mape*100,Et_tree_rmse,Et_tree_rmsle*100]
print()
print()
print()
print("####################################################################################################################")
print(                                        "# AdaBoostRegressor(ExtraTreeRegressor) #"                                   )
print("####################################################################################################################")

print("mae: "+str(ada_et_tree_mae))
print("mape: "+str(ada_et_tree_mape*100)+" %")
print("rmse: "+str(ada_et_tree_rmse))
print("rmsle: "+str(ada_et_tree_rmsle*100)+" %")
Ada_et_tree_list = [ada_et_tree_mae,ada_et_tree_mape*100,ada_et_tree_rmse,ada_et_tree_rmsle*100]
print()
print()
print()
print("####################################################################################################################")
print(                                          "# Deep Neural Network #"                                                   )
print("####################################################################################################################")

print("mae: "+str(deeplearning_model_mae))
print("mape: "+str(deeplearning_model_mape*100)+" %")
print("rmse: "+str(deeplearning_model_rmse))
print("rmsle: "+str(deeplearning_model_rmsle*100)+" %")
DNN_list = [deeplearning_model_mae,deeplearning_model_mape*100,deeplearning_model_rmse,deeplearning_model_rmsle*100]
print()
print()
print()



list=[lin_mape,
ridge_reg_mape,
lasso_reg_mape,
SVR_mape,
svm_rbf_reg_mape,
RandomForest_mape,
Et_tree_mape,
tree_mape,
ada_et_tree_mape,
deeplearning_model_mape]
list=[list[i]*100 for i in range(len(list))]

year=["LinearRegression",
"Ridge",
"Lasso",
"SVM(liner)",
"SVM(rbf)",
"Random_Forest",
"ExtraTree",
"DecisionTree",
"AdaBoost",
"DNN"]



# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
# plt.rcParams["font.family"] = "Palatino"
plt.figure(figsize=(30,13))
plt.xlabel("Model", fontsize=16)
plt.ylabel("MAPE(%)", fontsize=16)
plt.ylim(1, 50)
pop=list
plt.title('Test results(MAPE)', fontsize=17)
my_y_ticks = np.arange(0, 50, 5)
plt.yticks(my_y_ticks)
plt.plot(year, pop, color='y')
plt.scatter(year, pop, c='#DB7093', marker='s', alpha=0.9, s=10)
plt.bar(year, pop, color='#87CEFA', alpha=0.6)
for a, b in zip(year, pop):
    plt.text(a, b+1, "%.2f" % b, ha='center', va='bottom')

plt.axhline(y=15, label='y=15%', color='r', linestyle='--', linewidth=1.5)
plt.show()
