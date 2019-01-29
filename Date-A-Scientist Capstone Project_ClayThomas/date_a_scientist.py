import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from datetime import datetime

#/Users/clayborneo/Desktop/capstone_starter/dating_skeleton.py

# Create your df here:
df = pd.read_csv("/Users/clayborneo/Desktop/capstone_starter/profiles.csv")
#df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# Visualizing / Exploring the Data
#plt.hist(df.age, bins=20)
#plt.xlabel("Age")
#plt.ylabel("Frequency")
#plt.xlim(16, 80)
#plt.show()

#plt.hist(df.height, bins=20)
#plt.xlabel("Height")
#plt.ylabel("Frequency")
#plt.xlim(50, 90)
#plt.show()

#df.religion.head()
#df.religion.value_counts()

# Formulate a Question
# Can dataset features help us predict whether or not a person is agnostic/atheist?

#Augment the Data
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)

drug_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drug_mapping)

smoke_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes": 4}
df["smokes_code"] = df.smokes.map(smoke_mapping)

diet_mapping = {"mostly anything": 0, "anything": 1, "strictly anything": 2, "mostly vegetarian": 3, "mostly other": 4, "strictly vegetarian": 5, "vegetarian": 6, "strictly other": 7, "mostly vegan": 8, "other": 9, "strictly vegan": 10, "vegan": 11, "mostly kosher": 12, "mostly halal": 13, "strictly halal": 14, "strictly kosher": 15, "halal": 16, "kosher": 17}
df["diet_code"] = df.diet.map(diet_mapping)

education_mapping = {"graduated from college/university": 0, "graduated from masters program": 1, "working on college/university": 2, "working on masters program": 3, "graduated from two-year college": 4, "graduated from high school": 5, "graduated from ph.d program": 6, "graduated from law school": 7, "working on two-year college": 8, "dropped out of college/university": 9, "working on ph.d program": 10, "college/university": 11, "graduated from space camp": 12, "dropped out of space camp": 13, "graduated from med school": 14, "working on space camp": 15, "working on law school": 16, "two-year college": 17, "working on med school": 18, "dropped out of two-year college": 19, "dropped out of masters program": 20, "masters program": 21, "dropped out of ph.d program": 22, "dropped out of high school": 23, "high school": 24, "working on high school": 25, "space camp": 26, "ph.d program": 27, "law school": 28, "dropped out of law school": 29, "dropped out of med school": 30, "med school": 31}
df["education_code"] = df.education.map(education_mapping)

religion_mapping = {"agnosticism": 1, "other": 1, "agnosticism and laughing about it": 1, "atheism": 1, "other and laughing about it": 1, "atheism and laughing about it": 1, "catholicism and laughing about it": 0, "judaism and laughing about it": 0, "agnosticism but not too serious about it": 1, "catholicism but not too serious about it": 0, "christianity but not too serious about it": 0, "other but not too serious about it": 1, "judaism but not too serious about it": 0, "atheism but not too serious about it": 1, "buddhism but not too serious about it": 0, "christianity": 0, "catholicism": 0, "judaism": 0, "buddhism": 0, "hinduism": 0, "islam": 0, "christianity and somewhat serious about it": 0, "atheism and somewhat serious about it": 1, "other and somewhat serious about it": 1, "agnosticism and somewhat serious about it": 1, "christianity and very serious about it": 0, "atheism and very serious about it": 1, "catholicism and somewhat serious about it": 0, "other and very serious about it": 1, "buddhism and laughing about it": 0, "christianity and laughing about it": 0, "buddhism and somewhat serious about it": 0, "agnosticism and very serious about it": 1, "judaism and somewhat serious about it": 0, "hinduism but not too serious about it": 0, "catholicism and very serious about it": 0, "buddhism and very serious about it": 0, "hinduism and somewhat serious about it": 0, "hinduism and laughing about it": 0, "islam but not too serious about it": 0, "islam and somewhat serious about it": 0, "judaism and very serious about it": 0, "islam and laughing about it": 0,"hinduism and very serious about it": 0, "islam and very serious about it": 0}
df["religion_code"] = df.religion.map(religion_mapping)

#print(df.religion_code.value_counts())

# Normalize the Data
#feature_data = df[['drinks_code', 'drugs_code', 'smokes_code', 'diet_code', 'education_code', 'age']]
feature_data = df[['drugs_code', 'education_code', 'age']]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

# Remove Nans
feature_data_step = np.array(feature_data[~np.isnan(feature_data).any(axis=1)])
feature_data_clean = feature_data_step[0:20000, :]

prediction_data = df[['religion_code']]
prediction_data_step = np.array(prediction_data[~np.isnan(prediction_data).any(axis=1)])
prediction_data_clean = prediction_data_step[0:20000, :]

### Classification Techniques

x_train, x_test, y_train, y_test = train_test_split(feature_data_clean, prediction_data_clean, test_size=0.30, random_state=42)

# Naive Bayes Classifier
start_time = datetime.now()

nbc = MultinomialNB()

nbc.fit(x_train, y_train.ravel())
y_predict = nbc.predict(x_test)
y_predict_proba = nbc.predict_proba(x_test)
#print(y_predict)
#print(y_predict_proba)

# Accuracy / Precision / Recall

nbc_a_score = accuracy_score(y_test, y_predict)
nbc_p_score = precision_score(y_test, y_predict)
nbc_r_score = recall_score(y_test, y_predict)
nbc_f1_score = f1_score(y_test, y_predict)

#print(nbc_a_score)
#print(nbc_p_score)
#print(nbc_r_score)
#print(nbc_f1_score)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# K-Nearest Neighbors
start_time = datetime.now()

knn = KNeighborsClassifier(n_neighbors = 63)

knn.fit(x_train, y_train.ravel())
y_predict = knn.predict(x_test)
#print(y_predict)

# Accuracy / Precision / Recall

knn_a_score = accuracy_score(y_test, y_predict)
knn_p_score = precision_score(y_test, y_predict)
knn_r_score = recall_score(y_test, y_predict)
knn_f1_score = f1_score(y_test, y_predict)

#print(knn_a_score)
#print(knn_p_score)
#print(knn_r_score)
#print(knn_f1_score)

# Accuracy as K Changes
#k_list = list(range(1, 100))

# Create list to hold accuracy scores
#accuracy_scores = []

#for k in k_list:
	#classifier_accuracy = KNeighborsClassifier(n_neighbors=k)
	#classifier_accuracy.fit(x_train, y_train.ravel())
	#y_predict_accuracy = classifier_accuracy.predict(x_test)
	#scores = accuracy_score(y_test, y_predict_accuracy)
	#accuracy_scores.append(scores.mean())

#plt.plot(k_list, accuracy_scores)
#plt.xlabel('Number of Neighbors K')
#plt.ylabel('Accuracy Score')
#plt.show()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

### --------------------------------------------------------

### Regression Techniques

# K Neighbors Regressor

start_time = datetime.now()

knr = KNeighborsRegressor(n_neighbors = 63, weights = "distance")

knr.fit(x_train, y_train.ravel())
y_predict = knr.predict(x_test)

# Accuracy / Precision / Recall

knr_a_score = accuracy_score(y_test, y_predict.round())
knr_p_score = precision_score(y_test, y_predict.round())
knr_r_score = recall_score(y_test, y_predict.round())
knr_f1_score = f1_score(y_test, y_predict.round())

#print(knr_a_score)
#print(knr_p_score)
#print(knr_r_score)
#print(knr_f1_score)

# Accuracy as K Changes
#k_list = list(range(1, 100))

# Create list to hold accuracy scores
#accuracy_scores = []

#for k in k_list:
	#classifier_accuracy = KNeighborsRegressor(n_neighbors=k)
	#classifier_accuracy.fit(x_train, y_train.ravel())
	#y_predict_accuracy = classifier_accuracy.predict(x_test)
	#scores = accuracy_score(y_test, y_predict_accuracy.round())
	#accuracy_scores.append(scores.mean())

#plt.plot(k_list, accuracy_scores)
#plt.xlabel('Number of Neighbors K')
#plt.ylabel('Accuracy Score')
#plt.show()

# Precision as K Changes
#k_list = list(range(1, 100))

# Create list to hold precision scores
#precision_scores = []

#for k in k_list:
	#classifier_precision = KNeighborsRegressor(n_neighbors=k)
	#classifier_precision.fit(x_train, y_train.ravel())
	#y_predict_precision = classifier_precision.predict(x_test)
	#scores = precision_score(y_test, y_predict_precision.round())
	#precision_scores.append(scores.mean())

#plt.plot(k_list, precision_scores)
#plt.xlabel('Number of Neighbors K')
#plt.ylabel('Precision Score')
#plt.show()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# Multiple Linear Regression

start_time = datetime.now()

mlr = LinearRegression()

mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)
#print(y_predict)

# Accuracy / Precision / Recall

mlr_a_score = accuracy_score(y_test, y_predict.round())
mlr_p_score = precision_score(y_test, y_predict.round())
mlr_r_score = recall_score(y_test, y_predict.round())
mlr_f1_score = f1_score(y_test, y_predict.round())

#print(mlr_a_score)
#print(mlr_p_score)
#print(mlr_r_score)
#print(mlr_f1_score)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
