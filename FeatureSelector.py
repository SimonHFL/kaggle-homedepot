from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt


def check(df, predictors, target):
    
    # Perform feature selection
    selector = SelectKBest(f_classif, k=2)
    selector.fit(df[predictors], df[target])

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = selector.pvalues_

    # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='horizontal')
    plt.show()
