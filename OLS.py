import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from DataHelper import DataHandler
from LogisticRegression import LogRegress


class OLS:

    def __init__(self):
        self.dh = DataHandler()
        self.lr = LogRegress()
        return

    # ordinary least squares linear regression model
    def OLS(self, x, y):

        # setting up the score reports
        RSS_array = []
        RSS_reg_array = []
        MSE_array = []
        MSE_reg_array = []

        # initializing the lambda value for regularization
        lambda_value = 0.001

        # adding weights to the matrix and converting to 2D arrays
        x.insert(0, 'weight', [1 for _ in range(len(x))])
        x_array, y_array = np.array(x), np.array(y)

        # splitting data into 5 folds of data and calculating the data's linear regression
        # and assessing the final results with residual sum of squares (RSS) and mean sum squares (MSE)
        # once complete, averages are presented to reflecting the overall accuracy/behavior
        self.dh.div("Ordinary Least Squares, lambda = " + str(lambda_value))
        kf = KFold(n_splits=5, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x_array[train_index], x_array[test_index]
            y_train, y_test = y_array[train_index], y_array[test_index]
            w = np.dot(np.linalg.pinv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train.T))
            print("\nNumber of parameters :: {}".format(len(w)))
            y_pred = np.dot(w, x_test.T)
            RSS = np.sum((y_pred - y_test) ** 2) ** 0.5
            RSS_array.append(RSS)
            print("RSS is {:.2f}".format(RSS))

            MSE = mean_squared_error(y_test, y_pred)
            MSE_array.append(MSE)
            print("MSE is {:.2f}".format(MSE))

            reg_w = np.dot(np.linalg.pinv(np.dot(x_train.T, x_train) + lambda_value * np.identity(len(w))),
                           np.dot(x_train.T, y_train))
            reg_y_pred = np.dot(reg_w, x_test.T)

            RSS_reg = (np.sum((reg_y_pred - y_test) ** 2) + lambda_value * np.dot(reg_w.T, reg_w)) ** 0.5
            RSS_reg_array.append(RSS_reg)
            print("Regularized RSS is {:.2f}".format(RSS_reg))

            MSE_reg = mean_squared_error(y_test, reg_y_pred)
            MSE_reg_array.append(MSE_reg)
            print("Regularized MSE is {:.2f}".format(MSE_reg))

        self.dh.div("Average RSS & MSE")

        RSS_average = sum(RSS_array) / 5.0
        print("\nRSS average is {:.2f}".format(RSS_average))

        Reg_RSS_average = sum(RSS_reg_array) / 5.0
        print("Regularized RSS average is {:.2f}".format(Reg_RSS_average))

        MSE_average = sum(MSE_array) / 5.0
        print("\nMSE average is {:.2f}".format(MSE_average))

        MSE_RSS_average = sum(MSE_reg_array) / 5.0
        print("Regularized MSE average is {:.2f}".format(MSE_RSS_average))

    def featureCombos(self, dataset):

        # from the correlative findings discovered from the Pearson Correlation
        # the features concerning attack value, defense value are positively correlated
        # features to be tested in unison. On the other hand, negatively correlated features
        # such as capture rate and defense values we be tested in unison as well
        X, Y = self.dh.filterXYSets(dataset, ['attack_value', 'defense_value', 'stamina'])
        self.OLS(X, Y)

        X, Y = self.dh.filterXYSets(dataset, ['capture_rate', 'attack_value', 'defense_value'])
        self.OLS(X, Y)

        # turning the prediction methods around, we can attempt to predict highly
        # correlative features such as attack_value, defense_value, and stamina if we use
        # combat_point as a feature
        X, Y = self.dh.filterXYSets(dataset, ['defense_value', 'combat_point'], 'attack_value')
        self.OLS(X, Y)

        # Finally, pair uncorrelated features provide the opportunity to assess their behavior
        # with respect to error rates and accuracy
        X, Y = self.dh.filterXYSets(dataset, ['stamina', 'flee_rate ', 'spawn_chance'], 'combat_point')
        self.OLS(X, Y)

        X, Y = self.dh.filterXYSets(dataset, ['attack_value', 'flee_rate'], 'combat_point')
        self.OLS(X, Y)
