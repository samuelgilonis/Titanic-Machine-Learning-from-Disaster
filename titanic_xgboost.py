import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/samue/anaconda3/Library/bin/graphviz/'
from scipy.sparse import rand

# Define pre-processing measures
def pre_process(input_data):
    
    # Missing cabin info 
    input_data['Cabin'][input_data['Cabin'].isnull()] = "U"
    input_data['Cabin'][input_data['Cabin'] == 'T'] = "U"
    
    # # Extract a 'Deck' variable from 'Cabin'. Could also potentially take a room no. variable which might correspond to position fore/aft
    input_data['Deck'] = input_data['Cabin'].astype(str).str[0]
    
    # Impute missing embarkation points (modal values) - only two missing from training data
    input_data['Embarked'][input_data['Embarked'].isnull()] = "S"
    
    # Aggregate Parch and SibSp into one variable
    input_data['FamilySize'] = input_data['Parch'] + input_data['SibSp'] + 1
    input_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)
    input_data['FamilyType'] = ['Single' if family_size == 1 
                                else 'Small' if family_size < 5
                                else 'Large'\
                          for family_size in input_data['FamilySize']]
           
    # Extract titles from names - use REGEXP to take everything after ', ' and before '.'
    input_data['Title'] = input_data['Name'].str.extract(pat = '((?<=,\s)[a-zA-Z\s]+(?=\.))')
    input_data['Title'][input_data['Title'].isin(["Mlle","Ms"])] = "Miss"
    input_data['Title'][input_data['Title'].isin(["Mme"])] = "Mrs"
    input_data['Title'][input_data['Title'].isin(["Rev","Dr","Major","Col","Capt"])] = "Officer"
    input_data['Title'][input_data['Title'].isin(["Don","Lady","Sir","the Countess","Jonkheer"])] = "Royalty"
    ix = input_data['Title'].isin(['Mr','Mrs', 'Master', 'Miss', 'Officer', 'Royalty'])
    input_data['Title'][~ix] = "Other"
    
    # Impute missing fares (median values)
    fares = input_data.groupby(['Pclass', 'Embarked', 'FamilyType']).agg('median')['Fare']
    fares = fares.reset_index()
    def fill_fares(row):
        ix = (
            (fares['Pclass'] == row['Pclass']) &
            (fares['Embarked'] == row['Embarked']) &
            (fares['FamilyType'] == row['FamilyType'])
            )
        return fares['Fare'][ix].values[0]
    input_data['Fare'] = input_data.apply(lambda row: fill_fares(row) if np.isnan(row['Fare']) else row['Fare'], axis=1)
    
    # # Add log of fares
    input_data['Log(Fare)'] = np.log(input_data['Fare'])
    min_fare = input_data['Fare'][input_data['Fare'] != 0].min()
    input_data['Log(Fare)'][input_data['Log(Fare)'] == -np.inf] = np.log(min_fare)
    
    # Drop redundant variables
    input_data.drop(['PassengerId', 'Ticket', 'Age', 'Fare', 'Name', 'FamilySize', 'Cabin'], axis=1, inplace=True)
    
    # Label categorical variables and convert to dummies
    ohe = OneHotEncoder(drop='first')
    categorical_fields = list(input_data.select_dtypes(include=['object']).columns)
    categorical_fields.append('Pclass')
    categorical_data = input_data[categorical_fields]
    categorical_data = ohe.fit_transform(categorical_data).toarray() 
    headers = ohe.get_feature_names(categorical_fields)
    categorical_data = pd.DataFrame(categorical_data, columns = headers)
    input_data.drop(categorical_fields, axis=1, inplace=True)
    input_data = pd.concat([input_data, categorical_data], axis=1, sort=False)
    
    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    input_data[['Log(Fare)']] = sc.fit_transform(input_data[['Log(Fare)']])
    
    # Return output
    return input_data

# Import training data
filepath = "C:/Users/samue/Dropbox/Machine Learning/titanic/train.csv"
training_data = pd.read_csv(filepath)

# Pre-process data
training_data_processed = pre_process(training_data.copy())

# Fitting Decision Tree Classification to the Training set
X = training_data_processed.iloc[:,1:].values
y = training_data_processed.iloc[:,0].values

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# FIT MODELS

# Logistic regression
from sklearn.linear_model import LogisticRegression

def default_logistic(X_train, y_train, X_test):
    clf = LogisticRegression()      
    clf.fit(X_train, y_train)

    # make predictions for test data
    y_pred = clf.predict(X_test)
    
    return y_pred

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

def default_gauss_nb(X_train, y_train, X_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    
    return y_pred

def default_multinom_nb(X_train, y_train, X_test):
    gnb = MultinomialNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    return y_pred

def default_bernoulli_nb(X_train, y_train, X_test):
    gnb = BernoulliNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    return y_pred

# SVM
from sklearn.svm import SVC

def default_rbf_svm(X_train, y_train, X_test):
    clf = SVC()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    
    return y_pred

def default_linear_svm(X_train, y_train, X_test):
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    
    return y_pred

def default_poly_svm(X_train, y_train, X_test):
    clf = SVC(kernel='poly')
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    
    return y_pred

# Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def default_rf(X_train, y_train, X_test):
    clf = RandomForestClassifier()      
    clf.fit(X_train, y_train)

    # make predictions for test data
    y_pred = clf.predict(X_test)
    
    return y_pred

def default_extra_tree(X_train, y_train, X_test):
    clf = ExtraTreesClassifier()      
    clf.fit(X_train, y_train)

    # make predictions for test data
    y_pred = clf.predict(X_test)
    
    return y_pred

# Neural network
from sklearn.neural_network import MLPClassifier

def default_nn(X_train, y_train, X_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                         hidden_layer_sizes=(5, 2), random_state=10)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    
    return y_pred 

# Boosting
from xgboost import XGBClassifier

def default_xgboost(X_train, y_train, X_test):
    clf = XGBClassifier()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    
    return y_pred

# Create simple models 
def all_died(X_train, y_train, X_test):
    y_pred = np.zeros((len(X_test), 1))
    return y_pred

def percent_survived(X_train, y_train, X_test):
    prob_survival = 0.39
    y_pred = rand(len(X_test), 1, density=prob_survival, format='csr')
    y_pred.data[:] = 1
    y_pred = y_pred.toarray()
    return y_pred.astype(int)

def females_survive(X_train, y_train, X_test):
    col = training_data_processed.columns.get_loc('Sex_male')
    y_pred = 1 - X_test[:,col]
    return y_pred.astype(int)

def misters_die(X_train, y_train, X_test):
    col = training_data_processed.columns.get_loc('Title_Mr')
    y_pred = 1 - X_test[:,col]
    return y_pred.astype(int)

classifiers = {"No survivors!": all_died,
               "39 % chance": percent_survived,
               "Women survive" : females_survive,
               "Misters die" : misters_die,
               "Logistic Regression" : default_logistic, 
               "Guassian Naive Bayes" : default_gauss_nb, 
               #"Multinominal Naive Bayes" : default_multinom_nb,
               "Bernoulli Naive Bayes" : default_bernoulli_nb,
               "RBF SVM" : default_rbf_svm,
               "Linear SVM" : default_linear_svm,
               "Polynomial SVM" : default_poly_svm,
               "Random Forest" : default_rf,
               "Extra Tree" : default_extra_tree,
               "Neural Network" : default_nn,
               "XGBoost": default_xgboost
              }

# Functions to test accuracy
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

model_performance = pd.DataFrame(columns=['Model', 'Accuracy', 'Brier score', 'Area under ROC curve'])

# Preliminary run of all models
for key, value in classifiers.items():
    y_pred = value(X_train, y_train, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    model_performance = model_performance.append({'Model': key, 'Accuracy': accuracy, 'Brier score': brier, 'Area under ROC curve': roc_auc}, ignore_index=True)
    
# The below algorithm has the highest accuracy, lowest Brier score and the highest area under the ROC curve
# This indicates that it is accurately indentifying survivors without false-positives and with appropriate probabilities.

# Fit chosen model
clf = XGBClassifier()  
clf.fit(X_train, y_train)
    
# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer

parameters = {'n_estimators': [750], #number of trees, change it to 1000 for better results
              'max_depth': [3],
              'min_child_weight': [1],
              'gamma': [0.0],
              'subsample':[0.6],
              'colsample_bytree': [0.6],
              'reg_alpha':[0.1],
              'learning_rate': [0.05] #so called `eta` value
              }

scorer = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
clf_cv = GridSearchCV(clf, 
                      parameters, 
                      n_jobs=-1, # number of jobs to run in parallel. -1 means all processors
                      cv=StratifiedKFold(n_splits=5, shuffle=True), 
                      scoring=scorer,
                      verbose=5, 
                      refit='Accuracy')

# Re-fit the chosen model to all of the training data
X_train_all = training_data_processed.iloc[:,1:].values
y_train = training_data_processed.iloc[:,0].values
clf_cv.fit(X_train_all, y_train)

# Import submission data
filepath = "C:/Users/samue/Dropbox/Machine Learning/titanic/test.csv"
submission_data = pd.read_csv(filepath)
submission_data_processed = pre_process(submission_data.copy())

# Predicting the test set results
X_sub = submission_data_processed.values
y_pred = clf_cv.predict(X_sub)
y_pred = pd.DataFrame(y_pred, columns = ['Survived']).astype(int)
submission = pd.concat([y_pred, submission_data], axis=1, sort=False)
    
# Submission
submission = submission[['PassengerId', 'Survived']]
submission.to_csv("C:/Users/samue/Dropbox/Machine Learning/titanic/XGBoost_submission.csv", index=0)

# #Evaluate model
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

# Also take results from crude models

# def females_survive(X_train, y_train, X_test):
#     y_pred = 1 - submission_data_processed['Sex_male'].values
#     return y_pred.astype(int)

# def misters_die(X_train, y_train, X_test):
#     y_pred = 1 - submission_data_processed['Title_Mr'].values
#     return y_pred.astype(int)

# women_survive = females_survive('','',X_sub)
# women_survive = pd.DataFrame(women_survive, columns = ['Survived']).astype(int)
# women_survive = pd.concat([women_survive, submission_data], axis=1, sort=False)
# women_survive = women_survive[['PassengerId', 'Survived']]

# percent_survive = percent_survived('','',X_sub)
# percent_survive = pd.DataFrame(percent_survive, columns = ['Survived']).astype(int)
# percent_survive = pd.concat([percent_survive, submission_data], axis=1, sort=False)
# percent_survive = percent_survive[['PassengerId', 'Survived']]

# no_survivors = all_died('','',X_sub)
# no_survivors = pd.DataFrame(no_survivors, columns = ['Survived']).astype(int)
# no_survivors = pd.concat([no_survivors, submission_data], axis=1, sort=False)
# no_survivors = no_survivors[['PassengerId', 'Survived']]

# mr_dies = misters_die('','',X_sub)
# mr_dies = pd.DataFrame(mr_dies, columns = ['Survived']).astype(int)
# mr_dies = pd.concat([mr_dies, submission_data], axis=1, sort=False)
# mr_dies = mr_dies[['PassengerId', 'Survived']]

# # accuracy = accuracy_score(y_test, percent_survive)
# # brier = brier_score_loss(y_test, y_pred)
# # roc_auc = roc_auc_score(y_test, y_pred)

# # Submissions
# women_survive.to_csv("C:/Users/samue/Dropbox/Machine Learning/titanic/women_survive.csv", index=0)
# percent_survive.to_csv("C:/Users/samue/Dropbox/Machine Learning/titanic/percent_survive.csv", index=0)
# no_survivors.to_csv("C:/Users/samue/Dropbox/Machine Learning/titanic/no_survivors.csv", index=0)
# mr_dies.to_csv("C:/Users/samue/Dropbox/Machine Learning/titanic/mr_dies.csv", index=0)