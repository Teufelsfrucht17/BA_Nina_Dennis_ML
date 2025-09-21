
import GloablVariableStorage
from DataPreperation import featuresplit, combine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score




def OLSRegression(ticker):
    X,Y = combine(ticker)


    X_train_OLS, X_test_OLS, Y_train_OLS, Y_test_OLS = train_test_split(X,Y, test_size=0.2, random_state=42)

    param_grid = {'fit_intercept': [True, False]}
    # Set model specs
    ols_model = LinearRegression()
    CV_olsmodel = GridSearchCV(estimator=ols_model, param_grid=param_grid, cv=10)
    CV_olsmodel.fit(X_train_OLS, Y_train_OLS)

    # Prediction and result
    y_train_pred = CV_olsmodel.predict(X_train_OLS)
    y_test_pred = CV_olsmodel.predict(X_test_OLS)

    r2_train = r2_score(Y_train_OLS, y_train_pred)
    r2_test = r2_score(Y_test_OLS, y_test_pred)

    print("OLS score =", r2_train)
    print("OLS score =", r2_test)




OLSRegression(GloablVariableStorage.ListofStock)