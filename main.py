import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
# Load the dataset
data = pd.read_csv(r'C:\Users\hp\Desktop\CD.csv', encoding='gb2312')

# Separate the data into features and labels
x = data.iloc[:, 1:6].values
y = data.iloc[:, 0].values 

# Split the dataset into training and testing sets
test_size = 0.4
seed = 88
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)

# Standardize the features
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

# Random Forest Classifier setup and training
rfc = RandomForestClassifier(random_state=10, max_depth=3, min_samples_leaf=1, criterion='entropy', n_estimators=100)
rfc = rfc.fit(x_train, y_train)

# Evaluate the models on the test set
score_c = clf.score(x_test, y_test)
score_r = rfc.score(x_test, y_test)
print("SingleTree:{}".format(score_c), "RandomForest:{}".format(score_r))

# Logistic Regression setup and training
LR = LogisticRegression(random_state=0)
LR.fit(x_train, y_train)

# K-Nearest Neighbors setup and training
KNN = KNeighborsClassifier(n_neighbors=6, algorithm='ball_tree')
KNN = KNN.fit(x_train, y_train)
# Gaussian Process Classifier setup and training
gpc = GaussianProcessClassifier(random_state=0)
gpc = gpc.fit(x_train, y_train)

# Evaluating the Gaussian Process Classifier using cross-validation
score_gpc1 = cross_val_score(gpc, x_train, y_train, cv=3)
gpc_train = score_gpc1.mean()
gpctrain = np.std(score_gpc1, ddof=1)
score_gpc2 = cross_val_score(gpc, x_test, y_test, cv=3)
gpc_test = score_gpc2.mean()
print("gpc_train:{}".format(gpc_train), "gpc_test:{}".format(gpc_test))

# Evaluating the K-Nearest Neighbors model using cross-validation
score_KNN1 = cross_val_score(KNN, x_train, y_train, cv=4)
KNN_train = score_KNN1.mean()
KNNtrain = np.std(score_KNN1, ddof=1)
score_KNN2 = cross_val_score(KNN, x_test, y_test, cv=2)
KNN_test = score_KNN2.mean()
print("KNN_train:{}".format(KNN_train), "KNN_test:{}".format(KNN_test))

# Evaluating the Random Forest model using cross-validation
score_rfc1 = cross_val_score(rfc, x_train, y_train, cv=6)
rfc_train = score_rfc1.mean()
np.std(score_rfc1, ddof=1)
score_rfc2 = cross_val_score(rfc, x_test, y_test, cv=4)
rfc_test = score_rfc2.mean()
print("RF_train:{}".format(rfc_train), "RF_test:{}".format(rfc_test))

# Evaluating the Logistic Regression model using cross-validation
score_LR1 = cross_val_score(LR, x_train, y_train, cv=3)
LR_train = score_LR1.mean()
LRtrain = np.std(score_LR1, ddof=1)
score_LR2 = cross_val_score(LR, x_test, y_test, cv=3)
LR_test = score_LR2.mean()
print("LR_train:{}".format(LR_train), "LR_test:{}".format(LR_test))