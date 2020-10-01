from scipy.stats import pearsonr


class PearsonsCoefficient:

    # calculating the linearity of two data sets based on Pearson's Coefficient
    def getcoeff(self, dataset1, dataset2):
        corr, _ = pearsonr(dataset1, dataset2)
        return corr
