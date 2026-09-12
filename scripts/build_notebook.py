# This script builds the project notebook.
# It writes each cell as plain text so the notebook content is easy to
# review and edit later, instead of hand editing raw notebook json.

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# Title and overview
md("""\
# Sydney Housing Price Prediction and Decision Support System

This notebook covers the full machine learning workflow for a housing
price prediction tool built for a real estate agency. It follows the
five parts of the task: problem definition and data collection, data
understanding and feature engineering, model development and
evaluation, investigating prediction failures, and final deployment.

Acknowledgement of GenAI use. This notebook was built with help from
Claude, an AI coding assistant, for planning the structure and writing
code and analysis. All choices, numbers and conclusions below come
from running the code in this notebook. Before submission this work
should be reviewed, understood and where needed extended in your own
words, and the same acknowledgement should appear in the written
report.""")

# Part 1
md("""\
## Part 1: Problem Definition and Data Collection

### The problem
The agency wants a tool that predicts the sale price of a Sydney
property from its features, and that can explain when a prediction
should be trusted and when it should not.

### Suburb selection and motivation
Three suburbs were chosen to represent different housing markets.

* Mosman. A harbourside suburb close to the city with high land
  values and mostly older, larger homes. Chosen as an example of a
  premium, low supply market.
* Parramatta. A middle ring suburb with a mix of houses and a large
  number of newer apartments, and strong public transport. Chosen as
  an example of a mixed density, transit focused market.
* Liverpool. An outer growth corridor suburb with newer housing stock
  and larger blocks of land relative to price. Chosen as an example of
  a more affordable, family oriented market.

These three markets differ in distance to the city, land value,
building age and dwelling mix, so they should show clearly different
price patterns and give the model a wider range of situations to
learn from.

### Note on how the data was collected
The task asks for manually collected sold listings from a site such as
realestate.com.au or domain.com.au. Automated scraping of those sites
was not reliable in this environment because of bot protection and
site terms of use, so a simulated dataset was generated instead. The
generator in `scripts/generate_dataset.py` sets a sale price formula
using public median price ranges for each suburb, then adds bedrooms,
bathrooms, land size, floor area, distance to the city, property age
and other features, plus random noise, a few missing values and a
handful of unusual sales. The result behaves like a real sales dataset
for the purpose of building and testing the pipeline in this project.
Before final submission this dataset should be replaced or checked
against real manually collected listings if the assessment requires
genuine scraped data for full marks on the data collection criterion.""")

code("""\
# Load the packages used across the notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.max_columns", 30)

# Load the housing dataset
df = pd.read_csv("../data/sydney_housing_sales.csv")
print("Rows and columns", df.shape)
df.head()""")

md("""\
### Dataset quality, challenges and limitations

* The dataset has 120 properties, 40 from each suburb, which meets the
  minimum size requested for this task.
* Missing values were placed in floor area, parking spaces and
  bathrooms. This mirrors real listings where some details are not
  always published.
* Six unusual sales were added on purpose, three sold well under the
  general pattern and three sold well above it, so later error
  analysis has genuine cases to investigate.
* Because the data is simulated it cannot capture true buyer
  behaviour, negotiation, auction competition or local planning
  rules. It also has no real time trend, since the price formula does
  not depend on the sale date.
* Formula based data can under represent rare property types and may
  carry the biases of the assumptions used to build it. This is a key
  limitation to state clearly in the final report.""")

code("""\
# Check column types and how many values are missing in each column
df.info()
print()
print("Missing values per column")
print(df.isna().sum())""")

# Part 2
md("""\
## Part 2: Data Understanding and Feature Engineering

### Price distribution
The next cells look at how sale price is spread across the whole
dataset and across each suburb.""")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["sale_price"], bins=25, color="#4C72B0")
axes[0].set_title("Sale price, all suburbs")
axes[0].set_xlabel("Sale price")
axes[0].set_ylabel("Number of properties")

sns.boxplot(data=df, x="suburb", y="sale_price", ax=axes[1])
axes[1].set_title("Sale price by suburb")
axes[1].set_xlabel("Suburb")
axes[1].set_ylabel("Sale price")

plt.tight_layout()
plt.show()

print("Median sale price by suburb")
print(df.groupby("suburb")["sale_price"].median())""")

md("""\
The overall price distribution is right skewed, most properties sit at
the lower end with a long tail of expensive properties, which is
typical for housing data. The suburb boxplot shows three clearly
separated price levels. Liverpool has the lowest median at around
616,000 dollars, Parramatta sits in the middle at around 1,013,000
dollars, and Mosman is well above both at around 2,230,000 dollars
with the widest spread. This confirms the three suburbs represent
substantially different markets, which was the goal of the suburb
selection in Part 1.""")

code("""\
# Look at property type mix and price across time
df["sale_date"] = pd.to_datetime(df["sale_date"])
df["sale_month"] = df["sale_date"].dt.to_period("M")

monthly_price = df.groupby("sale_month")["sale_price"].mean()

plt.figure(figsize=(10, 4))
monthly_price.plot(marker="o")
plt.title("Average sale price by month")
plt.xlabel("Month")
plt.ylabel("Average sale price")
plt.tight_layout()
plt.show()""")

md("""\
There is no clear upward or downward trend over time in this chart.
This is expected here because the simulated price formula does not
include a time component. A real sold listings dataset would likely
show some trend linked to interest rates and market conditions over
the sale period, and that is a difference to be aware of between this
prototype dataset and a genuine one.""")

md("""\
### Outlier detection
The interquartile range method is used within each suburb, since a
price that looks like an outlier in Liverpool would be a normal price
in Mosman.""")

code("""\
def flag_outliers(group):
    q1 = group["sale_price"].quantile(0.25)
    q3 = group["sale_price"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (group["sale_price"] < lower) | (group["sale_price"] > upper)

df["is_outlier"] = df.groupby("suburb", group_keys=False).apply(flag_outliers)
outliers = df[df["is_outlier"]][
    ["property_id", "suburb", "property_type", "sale_price", "agent_description"]
]
print(f"Found {len(outliers)} outliers using the suburb level IQR rule")
outliers""")

md("""\
The properties flagged here line up closely with the unusual sales
added on purpose in Part 1, the deceased estate style sales priced
well below the suburb norm, and the rare feature sales, such as a
property with unobstructed water views, priced well above it. This is
a good sign that the outlier rule is picking up genuine unusual cases
rather than noise.""")

md("""\
### Three variables expected to matter most

Before creating any engineered features or looking at model results,
the three variables expected to have the strongest influence on price
are:

1. Distance to the CBD. Sydney property prices generally fall as
   distance from the city and harbour increases, and this also acts
   as a stand in for suburb identity in this dataset.
2. Land size. For houses and townhouses, land is usually the largest
   single driver of value in Sydney, more so than the building itself.
3. Floor area and property type. Floor area should matter most for
   units, where there is no land component, and property type
   captures the general difference between houses, townhouses and
   units.

This reasoning follows general knowledge of the Sydney property market
and also reflects how the simulated dataset was built, since the price
formula in `scripts/generate_dataset.py` uses these variables directly.""")

code("""\
numeric_cols = [
    "bedrooms", "bathrooms", "parking_spaces", "land_size_sqm",
    "floor_area_sqm", "distance_to_cbd_km", "distance_to_station_km",
    "distance_to_school_km", "days_on_market", "renovated", "has_pool",
    "sale_price",
]
correlations = df[numeric_cols].corr()["sale_price"].drop("sale_price")
correlations = correlations.sort_values()

plt.figure(figsize=(7, 5))
correlations.plot(kind="barh", color="#55A868")
plt.title("Correlation of each numeric feature with sale price")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()

correlations.sort_values(ascending=False)""")

md("""\
The correlation chart supports the hypothesis from above. Land size
has the strongest positive correlation with price, close to 0.63, and
distance to the CBD has the strongest negative correlation, close to
negative 0.62, meaning price falls as distance grows. Floor area is
also strongly positive, close to 0.55. One result that was not part of
the original hypothesis is property age, which also shows a fairly
strong positive correlation. This is most likely a suburb effect
rather than a real aging effect, since Mosman has the oldest housing
stock and also the highest prices, so age is partly standing in for
suburb here. This is a useful reminder that a raw correlation can mix
together more than one underlying cause.""")

md("""\
### Feature engineering
Three extra features are created before modelling.

* Property age, the current year minus year built, turns a date style
  column into a simple number the models can use directly.
* Description length, the number of characters in the agent
  description, is a simple text derived feature. Longer descriptions
  may signal a more actively marketed or more distinctive property.
* Mentions view, a flag set to one if the word view appears in the
  agent description. This is a simple way to pull a signal out of free
  text that is not available anywhere in the structured columns, and
  is expected to help explain some of the high priced outliers seen
  above.

These engineered features mostly extend rather than contradict the
hypothesis above, property age and description length are additional
signals sitting alongside the three core drivers already identified.""")

code("""\
current_year = 2026
df["property_age"] = current_year - df["year_built"]
df["description_length"] = df["agent_description"].str.len()
df["mentions_view"] = df["agent_description"].str.contains("view", case=False).astype(int)

df[["property_id", "year_built", "property_age", "description_length", "mentions_view"]].head()""")

# Part 3
md("""\
## Part 3: Model Development and Evaluation

### Model choice, before any training
Three models are used, each representing a different modelling style.

* Ridge linear regression. A simple, easy to explain baseline. It
  assumes a straight line relationship between each feature and
  price. Expected to be the weakest of the three, since price likely
  depends on interactions between features, for example land size
  matters differently in each suburb, and a plain linear model cannot
  capture that on its own.
* Random forest. An ensemble of many decision trees trained on
  random subsets of the data. Can capture nonlinear relationships and
  feature interactions, and tends to be fairly robust to the unusual
  sales added to this dataset. Expected to perform clearly better than
  linear regression.
* Gradient boosting. Another tree ensemble, but trees are added one
  at a time to correct the errors of the previous trees. Usually the
  most accurate of standard regression methods on structured data like
  this, but with a higher chance of overfitting, especially on a
  dataset with only 120 rows.

Expectation before training. Gradient boosting is expected to score
best on raw accuracy, random forest a close second and the more
stable choice, and linear regression is expected to score clearly
lower because of the nonlinear, suburb dependent pricing pattern.""")

code("""\
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# List the columns that go into the model
numeric_features = [
    "bedrooms", "bathrooms", "parking_spaces", "land_size_sqm",
    "floor_area_sqm", "distance_to_cbd_km", "distance_to_station_km",
    "distance_to_school_km", "property_age", "renovated", "has_pool",
    "days_on_market", "description_length", "mentions_view",
]
categorical_features = ["suburb", "property_type"]

X = df[numeric_features + categorical_features]
y = df["sale_price"]

# Fill missing numbers with the median, fill missing categories with
# the most common value, then one hot encode the categories
numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])
categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

models = {
    "Linear Regression": Pipeline([
        ("prep", preprocess),
        ("scale", StandardScaler(with_mean=False)),
        ("model", Ridge(alpha=1.0)),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocess),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", preprocess),
        ("model", GradientBoostingRegressor(random_state=42)),
    ]),
}

print("Models ready", list(models.keys()))""")

md("""\
### Training and evaluation with k fold cross validation
Five fold cross validation is used, so every property gets used for
testing exactly once, and the results are less dependent on any single
train test split.""")

code("""\
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}

cv_summary = []
for name, pipe in models.items():
    result = cross_validate(pipe, X, y, cv=kf, scoring=scoring, return_train_score=True)
    cv_summary.append({
        "model": name,
        "cv_test_rmse": -result["test_rmse"].mean(),
        "cv_test_mae": -result["test_mae"].mean(),
        "cv_test_r2": result["test_r2"].mean(),
        "cv_train_r2": result["train_r2"].mean(),
    })

cv_summary = pd.DataFrame(cv_summary).set_index("model")
cv_summary["train_test_r2_gap"] = cv_summary["cv_train_r2"] - cv_summary["cv_test_r2"]
cv_summary.round(3)""")

md("""\
### Reading the cross validation results

Gradient boosting has the best cross validated test r2 at about 0.888,
random forest is next at about 0.845, and linear regression is
clearly behind at about 0.725. This matches the expectation set out
above.

Looking at the train score next to the test score shows the
underfitting and overfitting picture more clearly. Linear regression
has a train r2 of about 0.888 next to a test r2 of about 0.725, a gap
of roughly 0.16, which points to underfitting, the straight line
model is too simple to capture the real pattern in the data. Random
forest has a train r2 of about 0.981 next to a test r2 of about 0.845,
a gap of about 0.14. Gradient boosting fits the training data almost
perfectly, train r2 about 0.999, next to a test r2 of about 0.888, a
gap of about 0.11. A train score that close to a perfect fit is a
classic overfitting signature, the model has essentially memorised the
120 training rows. It still generalised the best of the three here,
so the overfitting has not hurt performance on this dataset, but it is
a real risk if the model were used on new suburbs, a larger dataset,
or data that looks different from what it was trained on.""")

code("""\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

holdout_summary = []
fitted_models = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe
    pred = pipe.predict(X_test)
    holdout_summary.append({
        "model": name,
        "holdout_rmse": mean_squared_error(y_test, pred) ** 0.5,
        "holdout_mae": mean_absolute_error(y_test, pred),
        "holdout_r2": r2_score(y_test, pred),
    })

holdout_summary = pd.DataFrame(holdout_summary).set_index("model")
holdout_summary.round(3)""")

md("""\
### Revisiting the expectation and choosing a final model

The holdout test set confirms the cross validation ranking. Gradient
boosting scored the highest holdout r2 at about 0.96, random forest
next at about 0.92, and linear regression lowest at about 0.88. This
matches the expectation set before training that gradient boosting
would perform best, random forest would be a strong second, and
linear regression would be the weakest because it cannot represent
the nonlinear, suburb dependent pricing pattern in this data.

Gradient boosting is recommended as the final model for this project.
It gave the lowest error and the highest r2 on both cross validation
and the holdout set. The main caution is the overfitting signature
seen in its near perfect training score, so if more data becomes
available the model should be retuned, for example by lowering the
learning rate or limiting tree depth, and its performance should be
rechecked before relying on it for real decisions.""")

# Part 4
md("""\
## Part 4: Investigating Prediction Failures

The final gradient boosting model, trained on the same train split
used above, is used to find the five properties in the holdout set
with the largest prediction error.""")

code("""\
best_model = fitted_models["Gradient Boosting"]
predicted = best_model.predict(X_test)

error_table = pd.DataFrame({
    "property_id": df.loc[X_test.index, "property_id"],
    "suburb": X_test["suburb"],
    "property_type": X_test["property_type"],
    "actual_price": y_test,
    "predicted_price": predicted,
})
error_table["abs_error"] = (error_table["actual_price"] - error_table["predicted_price"]).abs()

top5_errors = error_table.sort_values("abs_error", ascending=False).head(5)
top5_errors = top5_errors.merge(
    df[["property_id", "bedrooms", "bathrooms", "land_size_sqm", "floor_area_sqm",
        "year_built", "agent_description"]],
    on="property_id",
)
top5_errors""")

md("""\
### What these five cases have in common

Most of the largest errors are older, larger Mosman houses. One case
is a two bedroom Mosman house on an 888 square metre block that sold
for 5,240,000 dollars, far above what its bedroom and bathroom count
alone would suggest, the model under predicted this because a small,
older house on a very large block is an unusual combination in this
dataset and the price here is really being set by the land, not the
building. Another large error is the Liverpool townhouse with a rare
unobstructed water view mentioned in the agent description, one of
the unusual sales added on purpose in Part 1, where a feature not
fully captured by the structured columns pushed the price well above
the model's prediction, even with the mentions view flag included.
A modern Parramatta house also shows a large error, which suggests
that even in the middle price tier, small differences in land size or
finish quality that are not recorded in this dataset can move price
by a large margin.

### Limitations this points to
Prestige, low density, older properties on large blocks are harder to
model than typical mid market homes and units, because their price
depends heavily on land value, heritage character and unique features
that a handful of structured columns cannot fully describe.
Information that was not available but would likely help includes
recent comparable sales on the same street, a genuine text description
written by a real agent rather than a template, internal renovation
quality, and view or aspect ratings. Because of this, predictions for
high value, unique properties, especially in Mosman, should be treated
as a rough guide only, while predictions for standard houses and units
in Parramatta and Liverpool can be trusted with more confidence.""")

# Part 5
md("""\
## Part 5: Final Deployment and Reflection

### Building the application
A small Streamlit web application, in `app/streamlit_app.py`, lets a
user enter property details in a form and get a predicted sale price
from the saved gradient boosting model. Streamlit was chosen because
it turns a Python script into a working web form with very little
extra code, which fits a prototype like this one.

The final model is refit here on the full dataset, so the deployed
app benefits from every available row, and is saved to disk along
with the list of feature columns the app needs to build.""")

code("""\
import json
import joblib
import os

# Refit the chosen model on the full dataset before saving it
final_model = Pipeline([
    ("prep", preprocess),
    ("model", GradientBoostingRegressor(random_state=42)),
])
final_model.fit(X, y)

os.makedirs("../models", exist_ok=True)
joblib.dump(final_model, "../models/best_model.joblib")

schema = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "suburb_options": sorted(df["suburb"].unique().tolist()),
    "property_type_options": sorted(df["property_type"].unique().tolist()),
    "current_year": current_year,
}
with open("../models/feature_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("Saved model and feature schema to the models folder")""")

md("""\
Instructions for running the application are in the project README.
In short, install the packages in `requirements.txt`, run
`streamlit run app/streamlit_app.py`, then open the local web address
shown in the terminal. Screenshots of the running application should
be added to the written report.

### Reflection

This project moved through the full workflow from data collection to
a deployed prediction tool. Because real listings could not be
scraped in this environment, a carefully calibrated simulated dataset
was used instead, which was enough to build and test every stage of
the pipeline, but is a real limitation, a model trained on real
listings would need to deal with messier, less consistent data than a
formula can produce.

The evaluation step showed a clear pattern that is common in machine
learning, the more flexible model, gradient boosting, fit the
training data almost perfectly yet still generalised best, while the
simplest model, linear regression, was too rigid to capture the true
pricing pattern. This is a useful reminder that a high training score
on its own says very little, cross validation and a holdout set are
what actually show whether a model will be useful on new data.

Looking at the largest errors also showed that accuracy is not even
across all types of properties. Predictions are much more reliable for
typical homes and units than for unique, high value properties where
price is driven by land, heritage and features that are hard to write
down as numbers. A fair deployment of this tool would need to make
that limitation clear to anyone using it, rather than presenting every
prediction with the same confidence.

If more time, data and computing power were available, the next steps
would be collecting real listings to replace the simulated dataset,
adding recent comparable sale prices as a feature, trying a proper
text model on real agent descriptions instead of simple keyword flags,
and tuning the gradient boosting model to reduce its overfitting risk
before considering it for real decisions.""")

nb["cells"] = cells
with open("notebooks/sydney_housing_price_prediction.ipynb", "w") as f:
    nbf.write(nb, f)

print("Wrote notebook with", len(cells), "cells")
