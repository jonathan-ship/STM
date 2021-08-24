|                                     | **2021-08-19** |                    |
|-------------------------------------|----------------|--------------------|
|                                     |                |                    |
| **Lead Time Prediction (Assembly)** |                |                    |
|                                     |                |                    |
|                                     |                | **생산공학연구실** |

**  
**

**Lead Time Prediction (Assembly)**

**Manual composition：**

-   **1. 리드타임 예측 모델 시스템 설명**

-   **2. 개발환경 및 실행 방법**

-   **3. Code설명 및 결과 출력**

    -   **데이터 가져오기**

    -   **머신 러닝 알고리즘을 위한 데이터 준비**

    -   **모델 선택과 훈련**

    -   **모델 세부 튜닝**

-   **4. 정확도 평가**

**  
**

-   **1. 리드타임 예측 모델 시스템 설명**

    -   **본 Manual에서Python 언어를 활용하여 데이터 분석을 수행하였으며 실제
        조선소 데이터에 학습 알고리즘을 적용하여 조립 공정의 생산 리드타임을
        예측해 보았다. 본 시험에서 분석에 활용되는 알고리즘은 크게 기계학습,
        심층학습, 앙상블학습이다.**

    -   **기계학습(지도학습) 알고리즘:**

        -   **Liner Regression**

        -   **Lasso**

        -   **Ridge**

        -   **Support Vector Machine**

            -   **선형 Support Vector Machine(LinearSVR)**

            -   **비선형 Support Vector Machine(KernelSVR)**

        -   **Decision tree**

    -   **심층학습 알고리즘:**

        -   **Deep learning**

    -   **앙상블 알고리즘:**

        -   **Random Forest**

        -   **Extra Trees**

        -   **Ada-boost**

**  
**

-   **2. 개발환경 및 실행 방법**

    -   **PyCharm**

        -   **가상 환경을 만든 다음에 pip를 이용해 아내의 package를**

            **다운로드 하기.**

| package      | version    |
|--------------|------------|
| python       | **3.6**    |
| pandas       | **1.1.5**  |
| scikit-learn | **0.22.2** |
| matplotlib   | **3.1.3**  |
| numpy        | **1.19.5** |
| keras        | **2.4.3**  |

-   **New project 만들기**

    -   **File New project Create**

        **![](media/4e40a0375c584c888e70d610184fb5c2.png)**

        -   **.py파일과 .csv파일 Location (project의 경로) 안에
            넣기![](media/6bc8c4b1b48865a239e81f67fa41f7a2.png)**

            -   **Block\_조립.py파일 실행**

                -   **Block\_조립.py**

                    ![](media/768c58b20115c3480441adffaf090347.png)

-   **3. Code설명 및 결과 출력**

    -   **.py파일 code 설명**

        -   **Import (python module)**

**![](media/d0731a63291bc2407ab14406748134ba.png)**

-   **실제 조선소의 데이터 파일(.csv)을 pandas (python중의 하나 library)로 읽어
    옴.**

    -   **조립\_리드타임\_실적.csv**

        **![](media/b5c9e4f1b909d839c1640fd057e0d4a0.png)**

-   **Ex: 출력 결과**

    **![](media/7867ae820a5c52dbbc1c6d7a0c796617.png)**

    -   **학습 데이터를 준비하기 위하여 전처리 과정 수행함**

        -   **데이터 전처리**

            **![](media/08e8540390dce987cd3916a48a547b5d.png)**

        -   **Feature, label의 dataframe를 따로 따로 뽑아 냄**

            **![](media/a97543d248bd041ebdd24eaa021d92f8.png)**

            -   **Feature 데이터에서 다시 법주형 데이터와 수치형 데이터를 구분이
                있게 dataframe를 따로 따로 뽑아 냄**

                **![](media/b45dcf8b2326d2faea753e0521543c08.png)**

                -   **Feature 데이터 중의 범주형 데이터에서 Scikit-learn
                    library가 제공한 Standardization 방법으로 전처리 수행**

                -   **Feature 데이터 중의 수치형 데이터에서 Scikit-learn
                    library가 제공한 OneHotEncode 방법으로 전처리 수행**

                    **![](media/3926fe13c2d29726bb595bd9e033e30c.png)**

**  
**

-   **Ex: 출력 결과**

    **![](media/37cf61b5a18ff429ac6112274dfa2f47.png)**

    -   **데이터 set는 8 :2의 비례로 training set, test set로 나눔**

        **![](media/fbe9e3c72c900e6a33c3c82d1f375b20.png)**

    -   **Training set를 이용하여 학습 모델을 훈련시킴**

        -   **Ex: Extra Trees**

            **![](media/5fe1ccffb55fd7ccfbb499f26927ea43.png)**

    -   **모델의 예측 정확도가 높이기 위하여(더 좋은 성능을 가지고 있는 모델을
        찾기 위하여) Scikit-learn library이 제공한 Grid Search, Randomized
        Search 방법을 이용하여 모델의 지정 parameter 범위 안에서 최적
        parameter를 찾는 과정 수행**

        **![](media/9d1b3e5dfea0dc493031e58ef0f251af.png)**

-   **출력 결과**

    **![](media/d62b3c2462f7cb541389bd203f8d9d62.png)**

    -   **최적 parameter를 찾고 나서 다시 모델 안에 넣고 마지막으로 test set에
        대하여 예측 수행**

        **![](media/e9b99ee30c1e31a2ca605481559283db.png)**

**  
**

-   **예측 평가지표**

    -   **MAE (mean absolute error)**

        -   **MAPE (mean squared percentage error)**

            -   **RMSE (root mean square error)**

            -   **RMSLE (root mean squared logarithmic error)**

                **![](media/5b8aa436fdfc509f2ed0474592af364c.png)**

-   **정확도 평가**

    **![](media/76742b5da0b3797a7c212025847b1ec3.png)**
