# === Standard Library ===
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline

from cleaning import load_and_clean_data

# Get cleaned dataset
master = load_and_clean_data()

master_ml = master[["room_type", "neighbourhood", "number_of_reviews", "reviews_per_month"]]
target = master["price"]
master_ml


# room_type and neighbourhood are both categorical variables, we therefore create dummies and drop the first colum every time, if all other columns are false it means the one remaining is true.
X = pd.get_dummies(master_ml, columns=["room_type", "neighbourhood"], drop_first=True)
y = target
display(X.shape)
display(y.shape)


# Here we split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Fit the linear regression model
linreg = LinearRegression(fit_intercept=True)
model = linreg.fit(X_train, y_train)

# calculate mse and mae
mse_test = mean_squared_error(y_test, linreg.predict(X_test))
mae_test = mean_absolute_error(y_test, linreg.predict(X_test))

print(f" The Mean Squared Error on the test set it {round(mse_test, 2)}, the mean absolute test error is {round(mae_test, 2)}")
print(f" The mean of the price is {round(y.mean(), 2)}")

# The models performance is not very good, on average the models
# estimations are off by 100.54 units almost 50% of the average value, the mean of 197.39

#Now testing if improved performance with lasso and ridge models

#Fit with lasso model
lasso = linear_model.Lasso(alpha=10, fit_intercept=True)
lasso.fit(X_train, y_train)

#calculate mse and mae
mse_test_lasso = mean_squared_error(y_test, lasso.predict(X_test))
mae_test_lasso = mean_absolute_error(y_test, lasso.predict(X_test))

print(f" With a lasso model The Mean Squared Error on the test set it {round(mse_test_lasso, 2)}, the mean absolute test error is {round(mae_test_lasso, 2)}")
print(f" The mean of the price is {round(y.mean(), 2)}")

# The lasso model isn't better than linear regression, the mean absolute error and mse remain quite large
# The model is rather unsatisfactory.

feature_names = X_train.columns

# Identify non-zero coefficients
selected_features = feature_names[lasso.coef_ != 0]
display(selected_features)



# Trying now something else to improve the model

# I'm creating a column called cool_words_points, this is key words that indicate that an 
# appartement could be expensive, I chose these by finding the most common words in the column name of
# the master dataframe and selecting a few that could indicate that an appartment would be more expensive
# depending on the amount of words present in name I allocate a higher score. I also use a quantile 
# transformer to smooth out and diminish impact of outliers and an elastic net combining both lasso and 
# ridge.
master = master.copy()

# Clean text from column name in master
master_description = master["name"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

# Split all words
all_words = " ".join(master_description).split()

# Count words
word_counts = Counter(all_words)

# Sort and display
word_freq_df = pd.DataFrame(word_counts.items(), columns=["word", "count"]).sort_values("count", ascending=False)

# from the dataframe above these are the words I selected
cool_words = ["center","central","spacious","penthouse","location", "heart", 
        "near", "location","located", "amazing"	, "parking", "serviced",
        "luxury", "crown", "design", "oasis", "lovely","lake", "view", "stylish",
        "downtown", "rooftop","garden", "beautiful", "bright", "private", "charming",
        "modern","centrally", "views", "schöne", "comfortable", "balkon", "zentrale", "lage"]

# creates this adds a point every time a word present in cool_words is present in name
master["cool_word_points"] = master["name"].str.lower().apply(
    lambda text: sum(word in text for word in cool_words)
)


# Creating my new design matrix with the new column cool_word_points
master_ml_words = master[["room_type", "neighbourhood", "number_of_reviews", "reviews_per_month", "cool_word_points"]]

y = master["price"]
X = pd.get_dummies(master_ml_words, columns=["room_type", "neighbourhood"], drop_first=True)

# I split the dataset here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compute Tukey fences on y_train:
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Make mask on so that I can then apply it to X_train
mask = (y_train >= lower) & (y_train <= upper)

# Apply mask to both X_train and y_train (matching indices)
X_train = X_train[mask]
y_train = y_train[mask]

cv = KFold(n_splits = 8, shuffle= True, random_state = 42)

# I run the elastic net model which use a combination of both lasso and ridge
elastic = ElasticNetCV(cv=cv)
elastic_fit = elastic.fit(X_train, y_train)
y_pred = elastic_fit.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))

# The model performs much better, maybe not so much because of the cool words but because of the 
# turkey fence that took out the greatest outliers, and maybe the elastics net.