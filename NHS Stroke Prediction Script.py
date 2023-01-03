# IMPORT LIBRARIES
import pandas as pd                 # For data analysis
import matplotlib.pyplot as plt     # For basic visualisation
import seaborn as sns               # For advanced visualisation
import numpy as np                  # For mathematical operations
import imblearn
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn import metrics
from sklearn import datasets
from sklearn import svm
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from numpy import median

# IMPORT DATA SET
df = pd.read_csv(
    r'C:\Users\jhews\OneDrive\Documents\Data Analytics\Big Data Applications\Assignments\Assignment 2\Data Set\healthcare-dataset-stroke-data.csv')
# View column titles
print(df.columns)
# View first 10 rows
print(df.head(10))

# DATA EXPLORATION & VISUALISATION
# Exploration and Visualisation of numeric variables:

# Age Stats Summary:
print(df['age'].describe())
# "Age" Histogram
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(df['age'], color=['#69b3a2'], edgecolor='black')
plt.xlabel("Age (Years)")
plt.ylabel("Frequency")
plt.title("Frequency of Patients by Age")
plt.show()

# "Age" Boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=df['age']).set(
    title='Boxplot showing summary of Age results')
plt.show()

# BMI Stats Summary:
print(df['bmi'].describe())
# "bmi" Histogram
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(df['bmi'])
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.title("Frequency of Patients by BMI results")
plt.show()

# "bmi" Boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=df['bmi']).set(
    title='Boxplot showing summary of BMI results')
plt.show()

# avg_glucose_level Stats Summary:
print(df['avg_glucose_level'].describe())
# "avg_glucose_level" Histogram
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(df['avg_glucose_level'])
plt.xlabel("Average Glucose Results")
plt.ylabel("Frequency")
plt.title("Frequency of Patients by Average Glucose Results")
plt.show()

# "avg_glucose_level" Boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=df['avg_glucose_level']).set(
    title='Boxplot showing summary of Average Glucose Levels results')
plt.show()


# Visualise categorical data

# Bar chart showing 'work_type'
fig = df['work_type'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Work Types')
plt.xlabel('Work Types')
plt.ylabel('Frequency of Work Type')
plt.show()
# Bar chart showing 'smoking_status'
fig = df['smoking_status'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Frequency of Smoking Status')
plt.show()
# Bar chart showing 'gender'
fig = df['gender'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency of Gender')
plt.show()
# Bar chart showing 'hypertension'
fig = df['hypertension'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Hypertension Status')
plt.xlabel('Hypertension Status')
plt.ylabel('Frequency of Hypertension Status')
plt.show()
# Bar chart showing 'heart_disease'
fig = df['heart_disease'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Heart Disease Status')
plt.xlabel('Heart Disease Status')
plt.ylabel('Frequency of Heart Disease Status')
plt.show()
# Bar chart showing 'ever_married'
fig = df['ever_married'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Frequency of Marital Status')
plt.show()
# Bar chart showing 'Residence_type'
fig = df['Residence_type'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Residence Type')
plt.xlabel('Residence Type')
plt.ylabel('Frequency of Residence Type')
plt.show()
# Bar chart showing 'stroke'
fig = df['stroke'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of Stroke Status')
plt.xlabel('Stroke Status')
plt.ylabel('Frequency of Stroke Status')
plt.show()

# DATA PREPREPARATION
# (1) Identify missing values

# Discover missing values
# Identify if there are any missing values in each variable
print(df.isnull().any())
print(df.isnull().sum())  # Count the sum of missing values in each variable
print('Row count is:', len(df.index))  # Count how many rows there are in total
# Calculate the % of rows missing in each variable to determine suitable data cleaning method
for column in df.columns:
    percentage = df[column].isnull().mean()
    print(f'{column}: {round(percentage*100, 2)}%')

# Visualise missing values on heat map
sns.heatmap(df.isnull(), cbar=False)
plt.title('Heatmap showing missing values in Stroke Dataset')
plt.show()

# Drop rows with missing values (only small percentage missing so delete missing values)
newdf = df.dropna()


# Check to ensure missing values are removed
# Visualise missing values on heat map
sns.heatmap(newdf.isnull(), cbar=False)
plt.title('Heatmap showing missing values in Stroke Dataset')
plt.show()
# Check new row count in comparison to original row count
print('Row count is:', len(newdf.index))
# Double check there are no rows with missing values
print(newdf.isnull().sum())


# (2) Identify Inconistent values
# Check categories within 'gender':
print(newdf['gender'].value_counts())

# Drop rows in 'gender' that contain "Other" classification
newdf = newdf[newdf['gender'].str.contains("Other") == False]

# Check "Other" classification in 'gender' has been removed:
print(newdf['gender'].value_counts())

# Convert all column headings to lower case:
newdf.columns = newdf.columns.str.lower()
print(newdf.columns)


# (3) Identify Outliers
# Visualise distribution
# Drop outliers

# Identified "bmi" outliers from Box Plot and drop outliers
newdf = newdf[(newdf['bmi'] < 46.2)]

# Check "bmi" outliers are dropped with histogram...
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(newdf['bmi'])
plt.title('Distribution of BMI results')
plt.xlabel('BMI')
plt.show()

# ...and then boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=newdf['bmi']).set(
    title='Boxplot showing summary of BMI results')
plt.show()

# Identified "avg_glucose_level" outliers and drop outliers
newdf = newdf[(newdf['avg_glucose_level'] <= 140)]

# Check "avg_glucose_level" outliers are dropped with histogram...
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(newdf['avg_glucose_level'])
plt.title('Distribution of Average Glucose Level results')
plt.xlabel('Average Glucose Levels')
plt.show()

# ...and then boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=newdf['avg_glucose_level']).set(
    title='Boxplot showing summary of Average Glucose Levels results')
plt.show()

# (4) Identify Duplications
duplicate = df[df.duplicated()]
print("Duplicate Rows :")
print(duplicate)


# CONVERT CATEGORICAL VARIABLES TO NUMBERIC:

# Use get_dummies() from pandas:

# Convert 'gender' to numeric using get_dummies:
gender_data = pd.get_dummies(newdf['gender'], drop_first=True)
# Check to see dummy variable has been created
print(gender_data.head(10))

# Convert 'ever_married' to numeric using get_dummies:
ever_married = pd.get_dummies(newdf['ever_married'], drop_first=True)
# Check to see dummy variable has been created
print(ever_married.head(10))

# Convert 'work_type' to numeric using get_dummies:
work_type = pd.get_dummies(newdf['work_type'], drop_first=True)
# Check to see dummy variable has been created
print(work_type.head(10))

# Convert 'residence_type' to numeric using get_dummies:
residence_type = pd.get_dummies(newdf['residence_type'], drop_first=True)
# Check to see dummy variable has been created
print(residence_type.head(10))

# Convert 'smoking_status' to numeric using get_dummies:
smoking_status = pd.get_dummies(newdf['smoking_status'], drop_first=True)
# Check to see dummy variable has been created
print(smoking_status.head(10))

# Concatenate dummy variables into original dataframe:
newdf = pd.concat([newdf, gender_data, ever_married,
                  work_type, residence_type, smoking_status], axis=1)
print(newdf.head(10))

# Drop original categorical variables from dataframe
newdf.drop(['gender', 'ever_married', 'work_type',
           'residence_type', 'smoking_status'], axis=1, inplace=True)
print(newdf.head(10))

# Rename column headings of dummy variables:
newdf.rename(columns={'Yes': 'ever_married', 'Never_worked': 'employment_never_worked', 'Private': 'employment_private_company', 'Self-employed': 'employment_self_employed',
                      'Urban': 'residence', 'formerly smoked': 'smoking_former_smoker', 'never smoked': 'smoking_never_smoked', 'smokes': 'smoking_smokes'}, inplace=True)
print(newdf.head(10))


# FEATURE SELECTION
# Correlogram: Visualise correlation of all variables with a heat map

cor = newdf.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(cor, cmap=plt.cm.CMRmap_r, annot=True)
plt.title("Heatmap Showing Correlation of Stroke Dataset")
plt.show()

# Correlation Analysis of 'age', 'hypertension' and 'heart_disease'
print("Correlation between 'age' and 'stroke': ",
      newdf['age'].corr(newdf['stroke']))
print("Correlation between 'hypertension' and 'stroke': ",
      newdf['hypertension'].corr(newdf['stroke']))
print("Correlation between 'heart_disease' and 'stroke': ",
      newdf['heart_disease'].corr(newdf['stroke']))


# Scatter plot with regression line of variables 'age' and 'stroke' to visualise correlation
sns.lmplot(x='age', y='stroke', data=newdf, x_jitter=0.1,
           y_jitter=0.1, line_kws={'color': 'red'})
plt.title('Scatter plot showing correlation between Stroke and Age')
plt.xlabel('Age (Years)')
plt.ylabel('Stroke')
plt.show()

# Scatter plot with regression line of variables 'hypertension' and 'stroke' to visualise correlation
sns.lmplot(x='hypertension', y='stroke', data=newdf, x_jitter=0.1,
           y_jitter=0.1, line_kws={'color': 'red'})
plt.title('Scatter plot showing correlation between Hypertension and Stroke')
plt.xlabel('Hypertension')
plt.ylabel('Stroke')
plt.show()

# Scatter plot with regression line of variables 'heart_disease' and 'stroke' to visualise correlation
sns.lmplot(x='heart_disease', y='stroke', data=newdf, x_jitter=0.1,
           y_jitter=0.1, line_kws={'color': 'red'})
plt.title('Scatter plot showing correlation between Heart Disease and Stroke')
plt.xlabel('Heart Disease')
plt.ylabel('Stroke')
plt.show()

# VIF test to check for multicollinearity between independent variables
# Create object "X" with independent variables
X = newdf[['heart_disease', 'hypertension', 'age']]
vif_data = pd.DataFrame()
vif_data["Independent Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_data)


# Identify dependent variable (y) and independent variables (X):
cols = ['heart_disease', 'hypertension', 'age']
x_data = newdf[cols]
y_data = newdf['stroke']


# REBALANCE CLASSIFICATION: SMOTE / RANDOM OVERSAMPLING

# (a) Random OverSampling:

# summarize class distribution
print(Counter(y_data))
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy=0.2)
# fit and apply the transform
X_over, y_over = oversample.fit_resample(x_data, y_data)
# summarize class distribution
print(Counter(y_over))

# (b) SMOTE

smote = SMOTE(random_state=42)
x_res, y_res = smote.fit_resample(x_data, y_data)
# summarize class distribution
print(Counter(y_res))

# Test (a) Random Over Sampling vs (b) SMOTE:
# Create Train and Test data (SMOTE)
# Determine 70/30 split between training and test data
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(
    x_res, y_res, test_size=0.7)

# Stroke Prediction Model (1) - Logistic  Regression
# Create Logistic Regression Model
model = LogisticRegression()

# Train Logistic Regression model
model.fit(x_training_data, y_training_data)

# Predict with Logistic Regression model
predictions = model.predict(x_test_data)

print(model.score(x_test_data, y_test_data))

# Measure performance of the model
print(classification_report(y_test_data, predictions))


# Create Train and Test data (Random Over Sampling)
# Determine 70/30 split between training and test data
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(
    X_over, y_over, test_size=0.7)


# MODEL SELECTION & IMPLEMENTATION

# Stroke Prediction Model (1) - LOGISTIC REGRESSION
# Create Logistic Regression Model
model = LogisticRegression()

# Train Logisitic Regression model
model.fit(x_training_data, y_training_data)

# Predict with Logisitic Regression model
predictions = model.predict(x_test_data)

# Measure performance of the model
# Classification Report
print(classification_report(y_test_data, predictions))
# Accuracy Score
print("Logistic Regression: Base Model Accuracy:",
      metrics.accuracy_score(y_test_data, predictions))
# Mean Absolute Error (MEA)
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test_data, predictions))
# Mean Squared Error (MSE)
print("Mean Squared Error (MSE):",
      metrics.mean_squared_error(y_test_data, predictions))
# Root Mean Squared Error (RMSE)
print("Root Mean Squared Error (RMSE):", np.sqrt(
    metrics.mean_squared_error(y_test_data, predictions)))

# Logistic Regression Hyperparameter Tuning:

# Ignore warnings
warnings.filterwarnings('ignore')

# Create parameter grid
parameters = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(0.001, -3, 3, 7),
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}

# Create GridSearchCV()
model = LogisticRegression()
clf = GridSearchCV(model,                     # model
                   param_grid=parameters,     # hyperparameters
                   scoring='accuracy',        # metric for scoring
                   cv=10)                     # number of folds
clf.fit(x_training_data, y_training_data)

# Print out tuned hyper parameters
print("Tuned Hyperparameters :", clf.best_params_)
# Print out Training Accuracy
print("Accuracy :", clf.best_score_)

# Build logistic model with tuned hyper parameters
model = LogisticRegression(C=0.1,
                           penalty='l2',
                           solver='newton-cg')
model.fit(x_training_data, y_training_data)
y_pred = model.predict(x_test_data)
print("Accuracy:", model.score(x_test_data, y_test_data))

# Create Confusion Matrix
lr_confusion_matrix = confusion_matrix(y_test_data, y_pred)
color = 'blue'
matrix = plot_confusion_matrix(
    model, x_test_data, y_test_data, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()


# Stroke Prediction Model (2) - RANDOM FOREST
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(x_training_data, y_training_data)
rf_predictions = random_forest.predict(x_test_data)
print("Random Forest Accuracy: ", accuracy_score(y_test_data, rf_predictions))
print(classification_report(y_test_data, rf_predictions))

# Create Random Hyper Parameter Grid:
# Number of trees in random forest
n_estimators = [200]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [50]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

# Identify best hyper parametres from Random Grid:
# Create Model:
random_forest = RandomForestClassifier()
# Random search of parameters, using 5 fold cross validation & 100 different combinations
random_forest_random = RandomizedSearchCV(
    estimator=random_forest,
    param_distributions=random_grid,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1)

# Fit the random search model
random_forest_random.fit(x_training_data, y_training_data)
rf_random_predictions = random_forest_random.predict(x_test_data)
# View best Parameters
print("Tuned Hyperparameters :", random_forest_random.best_params_)
# Ignore warnings
warnings.filterwarnings('ignore')
# Compare Random grid model to original model:
print(classification_report(y_test_data, rf_predictions))
print(classification_report(y_test_data, rf_random_predictions))

# Create the parameter grid
rf_parameters = {
    'bootstrap': [True],
    'max_depth': [50],
    'max_features': ['auto'],
    'min_samples_leaf': [1],
    'min_samples_split': [2, 4],
    'n_estimators': [200]
}
# Create a based model
random_forest = RandomForestClassifier()
rf_grid_search = GridSearchCV(random_forest,                # model
                              param_grid=rf_parameters,     # hyperparameters
                              scoring='accuracy',           # metric for scoring
                              cv=5)                         # number of folds

rf_grid_search.fit(x_training_data, y_training_data)

# Print out tuned hyper parameters
print("Tuned Hyperparameters :", rf_grid_search.best_params_)
# Print out Training Accuracy _ This one works better than original and random (0.87) - again you will have to adjust to the tuned hyperparameters set out in the random grid.
print("Accuracy :", rf_grid_search.best_score_)

# Visualise Random Forest Decision Tree

rf_small = RandomForestClassifier(n_estimators=10, max_depth=3)
rf_small.fit(x_training_data, y_training_data)

fig = plt.figure(figsize=(15, 10))
plot_tree(rf_small.estimators_[0],
          feature_names=cols,
          class_names='stroke',
          filled=True, impurity=True,
          rounded=True)
plt.show()

# Create Confusion Matrix
lr_confusion_matrix = confusion_matrix(y_test_data, rf_predictions)
color = 'blue'
matrix = plot_confusion_matrix(
    rf_grid_search, x_test_data, y_test_data, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()


# Stroke Prediction Model (3) - SUPPORT VECTOR MACHINE (SVM)

# Determine 70/30 split between training and test data
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(
    x_res, y_res, test_size=0.7)

svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(x_training_data, y_training_data)
svm_predictions = svm_classifier.predict(x_test_data)
print(classification_report(y_test_data, svm_predictions))
print("Accuracy:", metrics.accuracy_score(y_test_data, svm_predictions))

# Model tuning via Hyperparameters
# Check the default SVM hyperparameters
print(svm_classifier.get_params())
# Create Values for Hyperparameters

# Define the search space
parameters = {
    "C": [1.0],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ['linear', 'rbf']
}

# Define grid search
tuned_svm = GridSearchCV(svm_classifier,            # model
                         param_grid=parameters,     # hyperparameters
                         scoring='accuracy',        # metric for scoring,
                         cv=3)                      # number of folds

# Fit grid search
grid_result = tuned_svm.fit(x_training_data, y_training_data)
# Print out tuned hyper parameters
print("Tuned Hyperparameters :", tuned_svm.best_params_)
# Print out Training Accuracy
print("Accuracy :", tuned_svm.best_score_)

# Create Confusion Matrix
lr_confusion_matrix = confusion_matrix(y_test_data, svm_predictions)
color = 'blue'
matrix = plot_confusion_matrix(
    tuned_svm, x_test_data, y_test_data, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()
