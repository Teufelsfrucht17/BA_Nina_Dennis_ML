import pandas as pd


def createscore() -> pd.DataFrame:
    return pd.DataFrame(columns=["Model", "Sheet", "R2.Train", "R2.Test", "Parameter", "Notes"])


def _append(report: pd.DataFrame, addition: pd.DataFrame | None) -> pd.DataFrame:
    if addition is not None and not addition.empty:
        report = pd.concat([report, addition], ignore_index=True)
    return report


def runreports() -> pd.DataFrame:
    report = createscore()

    from OLS import runOLS
    report = _append(report, runOLS())

    from RidgeRegession import runRidgeRegession
    report = _append(report, runRidgeRegession())

    from RandomForest import Run_RandomForest
    report = _append(report, Run_RandomForest())

    from NeuralNetworksPytorch import runNN
    report = _append(report, runNN())

    return report


if __name__ == "__main__":
    df = runreports()
    df.to_excel("scoreModels.xlsx", index=False)
    print(df.head())
