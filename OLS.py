import pandas as pd

import DataPreperation
import Dataprep2
import GloablVariableStorage
from DataPreperation import featuresplit, combine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from createScoreModels import createscore


def OLSRegression(sheet:int,report:pd.DataFrame) -> pd.DataFrame:
    X,Y = Dataprep2.finalrunner(sheet)


    X_train_OLS, X_test_OLS, Y_train_OLS, Y_test_OLS = train_test_split(X,Y, test_size=0.2, random_state=42)

    param_grid = {'fit_intercept': [True, False]}
    # Set model specs
    ols_model = LinearRegression()
    CV_olsmodel = GridSearchCV(estimator=ols_model, param_grid=param_grid, cv=10,n_jobs=-1)
    CV_olsmodel.fit(X_train_OLS, Y_train_OLS)

    # Prediction and result
    y_train_pred = CV_olsmodel.predict(X_train_OLS)
    y_test_pred = CV_olsmodel.predict(X_test_OLS)

    r2_train = r2_score(Y_train_OLS, y_train_pred)
    r2_test = r2_score(Y_test_OLS, y_test_pred)

    print("Sheet:"+str(sheet)+" in-sample R2 =", r2_train) #in-sample R2
    print("Sheet:"+str(sheet)+" Out-of-sample R2 =", r2_test) #Out-of-sample R2

    report.loc[len(report)] = [
        "OLS",
        sheet,
        r2_train,
        r2_test,
        CV_olsmodel.best_params_['fit_intercept'],
        "N/A",
    ]
    return report


def runOLS() -> pd.DataFrame:

    report = createscore()

    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
            report = OLSRegression(i, report)
    except Exception as e:
        print(f"Ridge run failed: {e}")

    return report


runOLS()
