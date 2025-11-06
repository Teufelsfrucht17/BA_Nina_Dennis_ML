import pandas as pd


def createscore()-> pd.DataFrame:

    report = pd.DataFrame(columns=['Model',"Sheet",'R2.Train','R2.Test',"Parameter","commands"])

    return report


def runreports()-> pd.DataFrame:
    report = createscore()









    return report
