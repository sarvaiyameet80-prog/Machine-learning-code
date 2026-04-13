import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv(r"D:\meet\auto-mpg.csv")


df.replace("?", np.nan, inplace=True)


df["horsepower"] = pd.to_numeric(df["horsepower"])


df = df.dropna()


if "car name" in df.columns:
    df = df.drop(columns=["car name"])

   
    X = df.drop(columns=["mpg"])
    Y = df["mpg"]


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)



remaining_features = list(X.columns)
selected_features = []
best_score = -np.inf

print("Forward Feature Selection Process:\n")



while remaining_features:
    scores = []

   
    for feature in remaining_features:
        feature_to_test = selected_features + [feature]

       
        model = LinearRegression()
        model.fit(X_train[feature_to_test], Y_train)

        y_pred = model.predict(X_test[feature_to_test])
        score = r2_score(Y_test, y_pred)

        scores.append((score, feature))

    
    scores.sort(reverse=True)
    current_best_score, best_feature = scores[0]



    if current_best_score > best_score:
        best_score = current_best_score
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        print(f'Added: {best_feature}, R2 score: {best_score:.4f}')
    else:
        break

print('\n Final Selected features:')
print(selected_features)
