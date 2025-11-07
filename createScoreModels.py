import pandas as pd


def createscore() -> pd.DataFrame:
    return pd.DataFrame(columns=["Model", "Sheet", "R2.Train", "R2.Test", "Parameter", "Notes"])


def runreports() -> pd.DataFrame:
    from RidgeRegession import runRidgeRegession  # lazy import avoids circular dependency

    report = createscore()
    ridge_report = runRidgeRegession()
    report = pd.concat([report, ridge_report], ignore_index=True)
    return report


if __name__ == "__main__":
    print(runreports().head())
