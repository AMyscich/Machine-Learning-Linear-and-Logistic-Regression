from numpy import argmax
import pandas as pd

class OneHotEncoding:

    def __init__(self):
        return

    # encoding the primary strength (a categorical feature) to a numerically palatable feature
    def HotEncoding(self, dataset):

        # getting the unique encoded identifier for each primary strength and inserting columns initialized to 0
        primary_strength = dataset.primary_strength.unique()
        for ps in primary_strength:
            dataset.insert(len(dataset.columns) - 1, ps, [0 for _ in range(len(dataset))])

        # filling encoding columns with their respective values and dropping the primary_strength column
        for i, row in dataset.iterrows():
            dataset.loc[i, dataset.loc[i].primary_strength] = 1

        dataset = dataset.drop('primary_strength', axis=1)

        return dataset
