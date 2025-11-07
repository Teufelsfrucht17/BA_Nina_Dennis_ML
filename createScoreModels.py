import pandas as pd


def createscore() -> pd.DataFrame:
    return pd.DataFrame(columns=["Model", "Sheet", "R2.Train", "R2.Test", "Parameter", "Notes"])


def runreports() -> pd.DataFrame:
    report = createscore()

    from OLS import runOLS
    OLS = runOLS()
    report = pd.concat([report, OLS], ignore_index=True)

    from RidgeRegession import runRidgeRegession
    ridge_report = runRidgeRegession()
    report = pd.concat([report, ridge_report], ignore_index=True)

    from RandomForest import Run_RandomForest

    RF = Run_RandomForest()
    report = pd.concat([report, RF], ignore_index=True)
    return report


if __name__ == "__main__":
    print(runreports().head())
