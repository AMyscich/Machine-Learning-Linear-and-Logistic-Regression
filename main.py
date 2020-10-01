import pandas as pd

from DataHelper import DataHandler
from LogisticRegression import LogRegress
from OLS import OLS
from OneHotEncoding import OneHotEncoding


def main():
    dh = DataHandler()
    ohe = OneHotEncoding()
    ols = OLS()
    lr = LogRegress()

    dataframe = pd.read_csv("Data\\hw2_data.csv")

    dh.div("Question 1")
    dh.getColumnTypes(dataframe)

    dh.div("Question 2")
    dh.plotCorrelations(dataframe)

    dh.div("Question 3")
    dh.plotAllCorrelations(dataframe)

    dh.div("Question 4")
    dataframe = ohe.HotEncoding(dataframe)

    dh.div("Question 5, 6, & 7")
    x, y = dh.getXYSets(dataframe)
    ols.OLS(x, y)
    ols.featureCombos(dataframe)

    dh.div("Question 8")
    x_train, x_test, y_train, y_test = dh.SplitData(dataframe)
    print("Accuracy Rating :: {:.2f}%".format(lr.LogRegress(x_train, x_test, y_train, y_test)))

    dh.div("Question 9")
    lr.KFoldLogRegress(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
