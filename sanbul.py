import os
import tarfile
import urllib.request
import numpy as np

import pandas as pd

print("2016250056 황동근")

print("###############################################################")
print("Step1 : data preprocessing")
print("###############################################################")

print("")
print("1-1. Load Data from web server")
print("")
DOWNLOAD_ROOT = "YOUR_SERVER_URL"
FIRES_PATH = os.path.join("datasets", "sanbul")
FIRES_URL = DOWNLOAD_ROOT + "datasets/sanbul/sanbul-5.tgz"

def fetch_fires_data(fires_url=FIRES_URL, fires_path=FIRES_PATH):
    if not os.path.isdir(fires_path):
        os.makedirs(fires_path)
    tgz_path = os.path.join(fires_path, "sanbul-5.tgz")
    urllib.request.urlretrieve(fires_url, tgz_path)
    fires_tgz = tarfile.open(tgz_path)
    fires_tgz.extractall(path=fires_path)
    fires_tgz.close()

fetch_fires_data()

def load_fires_data(fires_path = FIRES_PATH):
    csv_path = os.path.join(fires_path, "sanbul-5.csv")
    return pd.read_csv(csv_path)

fires = load_fires_data()

fires_header = []
for col in fires:
    fires_header.append(col)

print("")
print("1-2. Print 2 fires.head(), fires.info(), fires.describe(), "
      "value_counts() about attributes of month, day")
print("")

print("fires.head() : ")
print(fires.head())
print("")

print("fires.info() : ")
print(fires.info())
print("")

print("fires.describe() : ")
print(fires.describe())
print("")

print('fires.value_counts("month") : ')
print(fires.value_counts("month"))
print("")

print('fires.value_counts("day") : ')
print(fires.value_counts("day"))
print("")

print("")
print("1-3. Visualized data")
print("")

import matplotlib.pyplot as plt
"""
A histogram displays numerical data 
by grouping data into "bins" of equal width

use hist in pandas dataframe 
"""
fires.hist(bins=50, figsize=(20,15))
plt.tight_layout()
plt.show()

print("")
print("1-4. Use log function improve distortion in burned_area")
print("")

from mpmath import ln

fires["burned_area"].hist(bins=50)
plt.title("burned_area")
plt.tight_layout()
plt.show()

"""
different from pdf
in pdf, y=ln(burned_area+1)
"""
def use_log(row):
    # return float(ln(row) + 5)         # for graph in pdf
    return float(ln(row + 1))

fires["burned_area"] = fires["burned_area"].apply(use_log)

fires["burned_area"].hist(bins=50)
plt.title("burned_area")
plt.tight_layout()
plt.show()

print("")
print("1-5. Split training/test set using train_test_split function"
      "from scikit-learn\n"
      "\t and check test set rate")
print("")

from sklearn.model_selection import train_test_split, GridSearchCV

train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

test_set.head()

fires["month"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

"""
proportion is rate
"""
print("\nMonth category proportion : \n",
      strat_test_set["month"].value_counts()/len(strat_test_set))

print("\nOverall month category proportion : \n",
      fires["month"].value_counts()/len(fires))

print("")
print("1-6. Print matrix more than four attributes using scatter_matrix() from pandas")
print("")

from pandas.plotting import scatter_matrix

"""
attribute of month and day don't plot
"""
attributes = ["longitude", "avg_temp", "max_wind_speed", "burned_area"]
scatter_matrix(fires[attributes], figsize=(12, 8))
plt.title("Scatter_matrix_plot")
plt.show()

print("")
print("1-7. Print matrix more than four attributes using scatter_matrix() from pandas"
      "\noption s : radius of circle"
      "\noption c : color")
print("")

fires.plot(kind="scatter", x="longitude", y="latitude",
           alpha=0.4, s=fires["max_temp"], label="max_temp",
           c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()

print("")
print("1-8. Print correlation matrix about target class(burned_area) using corr()")
print("")

corr_matrix = fires.corr()
print(corr_matrix["burned_area"].sort_values(ascending=False))

print("")
print("1-9. Encoding and print attributes of month and day using OneHotEncoder()")
print("")

"""
encoder is converting to numerical values from categorical values
example, "M, M, F, ... F" >> "1., 1., 0., ... 0"
"""
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

fires = strat_train_set.drop(["burned_area"], axis=1) # drop labels for training set
fires_labels = strat_train_set["burned_area"].copy()

encoded_month = encoder.fit_transform(fires[["month"]])
print("encoed_month : \n", encoded_month)
print("encoder.categories_ : \n", encoder.categories_)

print("")

encoded_day = encoder.fit_transform(fires[["day"]])
print("encoded_day : \n", encoded_day)
print("encoder.categories_ : \n", encoder.categories_)

fires_num = fires.drop(["month", "day"], axis=1)

print("")
print("1-10. Make encoded training set using Pipeline, StandardScaler from scikit-learn")
print("")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
('std_scaler', StandardScaler()),
])

num_attribs = list(fires_num)
cat_attribs = ["month", "day"]

full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OneHotEncoder(), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires)

print("###############################################################")
print("Step2 : model develop")
print("###############################################################")

print("")
print("2-1. Best model using GridSearchCV")
print("")

"""
it was not work, but its work now
"""
print("Stochastic Gradient Descent Regression using GridSearchCV")
print("in ch5")
from sklearn.linear_model import SGDRegressor
"""
source = "https://www.kaggle.com/nsrose7224/stochastic-gradient-descent-regressor"
"""
parms = {
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
}
sgd_reg = SGDRegressor()
sgd_reg.fit(fires_prepared, fires_labels)

grid_search_cv = GridSearchCV(sgd_reg, parms, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
sgd_best_model_cv = grid_search_cv.best_estimator_


"""
its work
"""
print("\nSupport Vector Machine using GridSearchCV")
print("in ch8")
from sklearn.svm import SVR
parms = {"kernel" : ["linear", "poly"],
         "C" : [0.1, 1, 100],
         "degree" : [2, 4],
         "epsilon" : [0.1, 1.0, 1.5]}

svm_reg = SVR()
svm_reg.fit(fires_prepared, fires_labels)

grid_search_cv = GridSearchCV(svm_reg, parms, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
svm_best_model_cv = grid_search_cv.best_estimator_


"""
its work
"""
print("\nDecision Tree using GridSearchCV")
print("in ch9")
from sklearn.tree import DecisionTreeRegressor
"""
source = "https://www.nbshare.io/notebook/312837011/Decision-Tree-Regression-With-Hyper-Parameter-Tuning-In-Python/"
"""
parms = {"splitter":["best","random"],
            "max_depth" : [1,9],
           "min_samples_leaf":[1,5,9],
           "max_features":["auto","sqrt",None],
           "max_leaf_nodes":[None, 40, 80] }

tree_reg = DecisionTreeRegressor()
tree_reg.fit(fires_prepared, fires_labels)

grid_search_cv = GridSearchCV(tree_reg, parms, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
tree_best_model_cv = grid_search_cv.best_estimator_


"""
its work
"""
print("\nRandom Forest using GridSearchCV")
print("in ch10")
from sklearn.ensemble import RandomForestRegressor
"""
source = "https://cyan91.tistory.com/18"
"""
parms = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
]

rnd_reg = RandomForestRegressor()
rnd_reg.fit(fires_prepared, fires_labels)

grid_search_cv = GridSearchCV(rnd_reg, parms, verbose=1, cv=10)
grid_search_cv.fit(fires_prepared, fires_labels)
rnd_reg_best_model_cv = grid_search_cv.best_estimator_



print("")
print("2-2. RMSE about training set using mean_squared_error")
print("")

from sklearn.metrics import mean_squared_error

# models = sgd_reg, svm_reg, tree_reg, rnd_reg

fires_predictions = sgd_best_model_cv.predict(fires_prepared)
sgd_mse = mean_squared_error(fires_labels, fires_predictions)
sgd_rmse = np.sqrt(sgd_mse)
sgd_rmse_reverted = np.exp(sgd_rmse) - 1
print("SGD - RMSE(training set) : \n",
      sgd_rmse_reverted)

fires_predictions = svm_best_model_cv.predict(fires_prepared)
svm_mse = mean_squared_error(fires_labels, fires_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse_reverted = np.exp(svm_rmse) - 1
print("SVM - RMSE(training set) : \n",
      svm_rmse_reverted)

fires_predictions = tree_best_model_cv.predict(fires_prepared)
tree_mse = mean_squared_error(fires_labels, fires_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse_reverted = np.exp(tree_rmse) - 1
print("DT - RMSE(training set) : \n",
      tree_rmse_reverted)

fires_predictions = rnd_reg_best_model_cv.predict(fires_prepared)
rnd_mse = mean_squared_error(fires_labels, fires_predictions)
rnd_rmse = np.sqrt(rnd_mse)
rnd_rmse_reverted = np.exp(rnd_rmse) - 1
print("RF - RMSE(training set) : \n",
      rnd_rmse_reverted)

print("")
print("2-3. draw learning curve grahp")
print("")

def plot_learning_curves(model, X, y):
 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
 train_errors, val_errors = [], []
 for m in range(1, len(X_train)):
    model.fit(X_train[:m], y_train[:m])
    y_train_predict = model.predict(X_train[:m])
    y_val_predict = model.predict(X_val)
    train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
    val_errors.append(mean_squared_error(y_val_predict, y_val))
 plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
 plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
 plt.legend(loc="upper right", fontsize=14) # not shown in the book
 plt.xlabel("Training set size", fontsize=14) # not shown
 plt.ylabel("RMSE", fontsize=14) # not shown

# _sgd_reg = SGDRegressor()
# _svm_reg = SVR()
# _tree_reg = DecisionTreeRegressor()
# _rnd_reg = RandomForestRegressor()

plot_learning_curves(sgd_best_model_cv, fires_prepared, fires_labels)
plt.show()

plot_learning_curves(svm_best_model_cv, fires_prepared, fires_labels)
plt.show()

plot_learning_curves(tree_best_model_cv, fires_prepared, fires_labels)
plt.show()

plot_learning_curves(rnd_reg_best_model_cv, fires_prepared, fires_labels)
plt.show()

print("")
print("2-4. print RMSE score using cross_val_score")
print("")

from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# models = sgd_reg, svm_reg, tree_reg, rnd_reg
sgd_score = cross_val_score(sgd_reg, fires_prepared,fires_labels,scoring="neg_mean_squared_error", cv=10)
svm_score = cross_val_score(svm_reg, fires_prepared,fires_labels,scoring="neg_mean_squared_error", cv=10)
tree_score = cross_val_score(tree_reg, fires_prepared,fires_labels,scoring="neg_mean_squared_error", cv=10)
rnd_score = cross_val_score(rnd_reg, fires_prepared,fires_labels,scoring="neg_mean_squared_error", cv=10)

sgd_rmse_score = np.sqrt(-sgd_score)
svm_rmse_score = np.sqrt(-svm_score)
tree_rmse_score = np.sqrt(-tree_score)
rnd_rmse_score = np.sqrt(-rnd_score)

print("\nSDG (train set): \n")
display_scores(sgd_rmse_score)
print("\nSVM (train set): \n")
display_scores(svm_rmse_score)
print("\nDT (train set): \n")
display_scores(tree_rmse_score)
print("\nRF (train set): \n")
display_scores(rnd_rmse_score)

print("")
print("2-5. RMSE about test set using mean_squared_error")
print("")

X_test = strat_test_set.drop(["burned_area"], axis=1) # drop labels for training set
Y_test = strat_test_set["burned_area"].copy()
X_test = full_pipeline.fit_transform(X_test)

sgd_score = cross_val_score(sgd_reg, X_test,Y_test,scoring="neg_mean_squared_error", cv=10)
svm_score = cross_val_score(svm_reg, X_test,Y_test,scoring="neg_mean_squared_error", cv=10)
tree_score = cross_val_score(tree_reg, X_test,Y_test,scoring="neg_mean_squared_error", cv=10)
rnd_score = cross_val_score(rnd_reg, X_test,Y_test,scoring="neg_mean_squared_error", cv=10)

sgd_rmse_score = np.sqrt(-sgd_score)
svm_rmse_score = np.sqrt(-svm_score)
tree_rmse_score = np.sqrt(-tree_score)
rnd_rmse_score = np.sqrt(-rnd_score)

print("\nSDG (train set): \n")
display_scores(sgd_rmse_score)
print("\nSVM (train set): \n")
display_scores(svm_rmse_score)
print("\nDT (train set): \n")
display_scores(tree_rmse_score)
print("\nRF (train set): \n")
display_scores(rnd_rmse_score)

print("###############################################################")
print("Keras model")
print("###############################################################")

fires = pd.read_csv("sanbul2district-divby100.csv")

train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

fires = train_set.drop(["burned_area"], axis=1) # drop labels for training set
fires_labels = train_set["burned_area"].copy()

fires_prepared = full_pipeline.fit_transform(fires)

import tensorflow as tf
from tensorflow import keras
X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)

X_test = test_set.drop(["burned_area"], axis=1) # drop labels for training set
Y_test = test_set["burned_area"].copy()
X_test = full_pipeline.fit_transform(X_test)

np.random.seed(42)
tf.random.set_seed(42)

mlp_model = keras.models.Sequential([
keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
keras.layers.Dense(1)
])
mlp_model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = mlp_model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))
mse_test = mlp_model.evaluate(X_test, Y_test)
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca()
plt.show()

model_version = "0001"
model_name = "my_fires_model"
model_path = os.path.join( model_name, model_version)
tf.saved_model.save( mlp_model, model_path)