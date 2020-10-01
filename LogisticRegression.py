import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from DataHelper import DataHandler


class LogRegress:

    def __init__(self):
        self.dh = DataHandler()
        return

    # performing logistic regression given training and test sets
    def LogRegress(self, x_train, x_test, y_train, y_test, reg_set=1e42):

        # initializing the logistic regression model to supports both L1 and L2 regularization, with a dual
        # formulation only for the L2 penalty. To prevent unwanted regularization, the C (or 1/lambda)
        # parameter is set to an abnormally high value
        logistic_regression = LogisticRegression(solver='liblinear', C=reg_set, max_iter=1000, random_state=0)
        logistic_regression.fit(x_train, y_train)
        y_pred = logistic_regression.predict(x_test)

        self.dh.div("Logistic Regression Accuracy, lambda = {:.4f}".format(1 / reg_set))

        # returning the accuracy comparison percentage between the predicted values and the test values
        return metrics.accuracy_score(y_test, y_pred)

    # Cross validating the training set into 5 folds where the best regularization coefficient will be selected
    def KFoldLogRegress(self, x_train, x_test, y_train, y_test, initial_lambda=0.0001):

        # default lambda coefficient is set to 0.0001 and will increase
        # by several orders of magnitude for each iteration over the folds
        target_reg_term = lambda_value = initial_lambda
        target_accuracy = 0
        x, y = np.array(x_train), np.array(y_train)

        # randomizing the folds into 5 separate sets for iterations
        kf = KFold(n_splits=5, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(x_train):
            kf_x_train, kf_x_test = x[train_index], x[test_index]
            kf_y_train, kf_y_test = y[train_index], y[test_index]

            # getting the accuracy rating from the Logistic Regression
            # functions and assessing the accuracy in the terminal
            results = self.LogRegress(kf_x_train, kf_x_test, kf_y_train, kf_y_test, (1 / lambda_value))
            print("Accuracy Rating :: {:.2f}%".format(results * 100))

            # keeping track of the best lambda with the highest accuracy rating
            if target_accuracy < results:
                print("Swapping for lambda = ", lambda_value)
                target_accuracy = results
                target_reg_term = lambda_value

            lambda_value *= 100

        # testing best regularization parameter for the test set
        print("Final Accuracy Rating :: {:.2f}%"
              .format(self.LogRegress(x_train, x_test, y_train, y_test, (1 / target_reg_term))*100))


