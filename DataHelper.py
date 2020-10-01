import seaborn as sns
import numpy as np
import itertools as it

from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
from PearsonCorrelation import PearsonsCoefficient


class DataHandler:

    def __init__(self):
        return

    def getColumnTypes(self, dataset):

        # getting all available columns in the data frame and the numeric columns
        cols = dataset.columns
        num_cols = dataset._get_numeric_data().columns

        # printing the numeric column names
        print("Numeric Column(s) :: ")
        for col in num_cols:
            print(col)

        # printing the categorical column names
        print("\nCategorical Column(s) :: ")
        cat_cols = list(set(cols) - set(num_cols))
        for col in cat_cols:
            print(col)

        return num_cols, cat_cols

    def plotCorrelations(self, dataset):

        # initializing the pair grid to assess a series of features compared to the combat points
        # minor features such as a length and aesthetic attributes are added to the plot design
        g = sns.PairGrid(dataset,
                         y_vars=["combat_point"],
                         x_vars=["stamina", "attack_value", "defense_value", "capture_rate", "flee_rate",
                                 "spawn_chance"])
        g.map(sns.scatterplot, color="Green", alpha=.7)
        g.add_legend()

        # dropping the categorical features, name and primary_strength
        dataset = dataset.drop(columns=['name', 'primary_strength'])

        # numerical feature comparisons using Pearson's Coefficient correlations
        pc = PearsonsCoefficient()
        for (columnName, columnData) in dataset.iteritems():
            if columnName != 'combat_point':
                print(columnName + " vs. combat points" + " :: {0:.4f}".format(
                    pc.getcoeff(columnData.values, dataset.loc[:, ['combat_point']].values.reshape(-1, ).tolist())))

        plt.tight_layout()
        plt.show()

    # plotting all combinations of relationships between each of the numeric features
    def plotAllCorrelations(self, dataset):

        # dropping the categorical features, name and primary_strength
        dataset = dataset.drop(columns=['name', 'primary_strength'])

        # initializing the pair grid to assess a series of features compared to the combat points
        # minor features such as a length and aesthetic attributes are added to the plot design
        g = sns.pairplot(dataset)
        g.map(sns.scatterplot, color="SeaGreen", alpha=.7, markers=["D"])
        g.add_legend()

        # numerical feature comparisons using Pearson's Coefficient correlations
        pc = PearsonsCoefficient()
        correlation_list = []
        for column in it.combinations(dataset.columns, 2):
            correlation_list.append([pc.getcoeff(dataset.loc[:, [column[0]]].values.reshape(-1, ).tolist(),
                                                 dataset.loc[:, [column[1]]].values.reshape(-1, ).tolist()), column[0],
                                     column[1]])

        correlation_list.sort()

        for correlation in reversed(correlation_list):
            print(correlation[1] + " vs. " + correlation[2] + " :: {0:.4f}".format(correlation[0]))

        plt.tight_layout()
        plt.show()

    def getXYSets(self, dataset):

        # dropping unnecessary features (name and combat_point) from data set X
        # On the other hand, data set Y only concerns the targeted output feature, combat_point
        y = dataset.combat_point
        x = dataset.drop(columns=['name', 'combat_point'])

        return x, y

    def addLogisticBoolean(self, dataset, column_name, split_value):

        # inserting a logistical True/False column to describe a mutually exclusive features,
        # given a comparison threshold to differentiate both
        dataset.insert(len(dataset.columns) - 1, column_name,
                       [True if split_value <= dataset.loc[i].combat_point else False for i in range(len(dataset))])

        return dataset

    def SplitData(self, dataset, random_state=None):

        # calculating the logistic threshold comparator for a combat point set
        mean = np.mean(np.array(dataset.combat_point))

        # adding the logistic boolean feature column
        dataset = self.addLogisticBoolean(dataset, 'high_combat_point', mean)

        # splitting data between training and test sets at a randomly shuffled 80%-20% training-testing ratio
        x_train, x_test, y_train, y_test = train_test_split(dataset, dataset.high_combat_point, test_size=0.2,
                                                            random_state=random_state)

        # removing unnecessary features in data set X
        x_train, x_test = x_train.drop(columns=['name', 'high_combat_point', 'combat_point']), \
                          x_test.drop(columns=['name', 'high_combat_point', 'combat_point'])

        return x_train, x_test, y_train, y_test

    def filterXYSets(self, dataset, featured_columns, target_column='combat_point'):

        # dropping unnecessary features (name and combat_point) from data set X
        # On the other hand, data set Y only concerns the targeted output feature, combat_point
        columns_to_remove = [col for col in dataset.columns.tolist() if col not in featured_columns]

        y = dataset[target_column]
        x = dataset.drop(columns=columns_to_remove)

        return x, y

    # pretty printing feature for terminal
    def div(self, string):
        print("\n========================= " + string + " =========================")
