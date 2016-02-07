import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import FeatureSelector
print(1)
df_train = pd.read_csv("files/train.csv", encoding="ISO-8859-1")
df_test = pd.read_csv("files/test.csv", encoding="ISO-8859-1")
descriptions = pd.read_csv("files/product_descriptions.csv", encoding="ISO-8859-1")

num_train = df_train.shape[0]

def common_words(str1, str2):
    word_count = 0
    words = str1.split()

    for word in words:
        if word in str2:
            word_count += 1

    return word_count

df = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df = pd.merge(df, descriptions, how='left', on='product_uid')
print(2)
df["word_in_title"] = df.apply(lambda row: common_words(row["search_term"], row["product_title"]), axis=1)
df["word_in_description"] = df.apply(lambda row: common_words(row["search_term"], row["product_description"]), axis=1)

train = df.iloc[:num_train]
test = df.iloc[num_train:]
print(3)
predictors = ["word_in_description", "word_in_title"]

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(train[predictors].values, train["relevance"].values)
predictions = clf.predict(test[predictors].values)

submission = pd.DataFrame({
        "id": test["id"],
        "relevance": predictions
    }).to_csv('submission.csv',index=False)

#FeatureSelector.check(train, predictors, "relevance")