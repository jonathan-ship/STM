[Lead Time prediction model manual.docx](https://github.com/jonathan-ship/STM/files/7028963/Lead.Time.prediction.model.manual.docx)
# STM
  ##1. 리드타임 예측 모델 시스템 설명
    본 Manual에서Python 언어를 활용하여 데이터 분석을 수행하였으며 실제 조선소 데이터에 학습 알고리즘을 적용하여 조립 공정의 생산 리드타임을 예측해 보았다. 본 시험에서 분석에 활용되는 알고리즘은 크게 기계학습, 심층학습, 앙상블학습이다.
    기계학습(지도학습) 알고리즘:
      *Liner Regression
      *Lasso
      *Ridge
      *Support Vector Machine 
  	선형 Support Vector Machine(LinearSVR)
  	비선형 Support Vector Machine(KernelSVR)
      *Decision tree
  	심층학습 알고리즘:
      *Deep learning
    앙상블 알고리즘:
      *Random Forest
      *Extra Trees
      *Ada-boost
  ##2. 개발환경 및 실행 방법
    PyCharm 
      가상 환경을 만든 다음에 pip를 이용해 아내의 package를 다운로드 하기.
      package	version
      *python	3.6
      *pandas	1.1.5
      *scikit-learn	0.22.2
      *matplotlib	3.1.3
      *numpy	1.19.5
      *keras	2.4.3
    New project 만들기
      File → New project → Create
      .py파일과 .csv파일 Location (project의 경로) 안에 넣기
      Block_조립.py파일 실행 
        Block_조립.py
  ##3. Code설명 및 결과 출력
    .py파일 code 설명 
      Import (python module)
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
    실제 조선소의 데이터 파일(.csv)을 pandas (python중의 하나 library)로 읽어 옴.
      조립_리드타임_실적.csv
    학습 데이터를 준비하기 위하여 전처리 과정 수행함
      Feature, label의 dataframe를 따로 따로 뽑아 냄
      Feature 데이터에서 다시 법주형 데이터와 수치형 데이터를 구분이 있게 dataframe를 따로 따로 뽑아 냄
        Feature 데이터 중의 범주형 데이터에서 Scikit-learn library가 제공한 Standardization 방법으로 전처리 수행
        Feature 데이터 중의 수치형 데이터에서 Scikit-learn library가 제공한 OneHotEncode 방법으로 전처리 수행
    데이터 set는 8 :2의 비례로 training set, test set로 나눔
    Training set를 이용하여 학습 모델을 훈련시킴
    모델의 예측 정확도가 높이기 위하여(더 좋은 성능을 가지고 있는 모델을 찾기 위하여) Scikit-learn library이 제공한 Grid Search, Randomized Search 방법을 이용하여 모델의 지정 parameter 범위 안에서 최적 parameter를 찾는 과정 수행
    최적 parameter를 찾고 나서 다시 모델 안에 넣고 마지막으로 test set에 대하여 예측 수행
  ##4.정확도 평가 
    예측 평가지표
      MAE (mean absolute error)
      MAPE (mean squared percentage error)
      RMSE (root mean square error)
      RMSLE (root mean squared logarithmic error)
