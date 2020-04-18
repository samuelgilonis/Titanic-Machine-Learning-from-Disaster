import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/samue/anaconda3/Library/bin/graphviz/'
from scipy.sparse import rand

# Define pre-processing measures
def pre_process(input_data):
    # Remove ticket number. N.B. This could potentially be used to identify co-travellers (look for proximate ticket no.s)
    input_data.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
    
    # # Missing cabin info could potentially be significant (could, e.g. correlate with Pclass)
    # input_data['Cabin'][input_data['Cabin'].isnull()] = "U"
    # input_data['Cabin'][input_data['Cabin'] == 'T'] = "U"
    
    # # Extract a 'Deck' variable from 'Cabin'. Could also potentially take a room no. variable which might correspond to position fore/aft
    # input_data['Deck'] = input_data['Cabin'].astype(str).str[0]
    input_data.drop(['Cabin'], axis=1, inplace=True)
    
    # Impute missing ages and fares (median values)
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
    imputer = imputer.fit(input_data[['Age', 'Fare']])
    input_data[['Age', 'Fare']] = imputer.transform(input_data[['Age', 'Fare']])
    
    # # Extract new categorical 'stage of life' variable
    # input_data['Stage'] = ["Baby" if age < 1 \
    #                      else "Child" if age < 13 \
    #                      else "Teenager" if age < 18 \
    #                      else "Adult" if age < 50 \
    #                      else "Elderly" \
    #                      for age in input_data['Age']] 
    
    # Impute missing embarkation points (modal values) - only two missing from training data
    input_data['Embarked'][input_data['Embarked'].isnull()] = "S"
    
    # Extract titles from names - use REGEXP to take everything after ', ' and before '.'
    input_data['Title'] = input_data['Name'].str.extract(pat = '((?<=,\s)[a-zA-Z]+(?=\.))')
    input_data['Title'][input_data['Title'].isin(["Mlle","Ms"])] = "Miss"
    input_data['Title'][input_data['Title'].isin(["Mme"])] = "Mrs"
    ix = input_data['Title'].isin(['Mr','Mrs', 'Master', 'Miss'])
    input_data['Title'][~ix] = "Exotic title"
    
    # Drop 'Name' field
    input_data.drop(['Name'], axis=1, inplace=True)
    
    # Aggregate Parch and SibSp into one variable
    input_data['FamilySize'] = input_data['Parch'] + input_data['SibSp']
    input_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)
    
    # # Extract new categorical 'is alone?' variable
    # input_data['IsAlone'] = [1 if family_size == 0 else 0 \
    #                          for family_size in input_data['FamilySize']]
    
    # Following fields can be made categorical: Pclass, Sex, Embarked, Deck, Title
    ohe = OneHotEncoder(drop='first')
    #for field in ['Pclass', 'Sex', 'Embarked', 'Deck', 'Title']:
    categorical_fields = list(input_data.select_dtypes(include=['object']).columns)
    categorical_fields.append('Pclass')
    categorical_data = input_data[categorical_fields]
    categorical_data = ohe.fit_transform(categorical_data).toarray() 
    headers = ohe.get_feature_names(categorical_fields)
    categorical_data = pd.DataFrame(categorical_data, columns = headers)
    input_data.drop(categorical_fields, axis=1, inplace=True)
    input_data = pd.concat([input_data, categorical_data], axis=1, sort=False)
    
    # Feature scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    input_data[['Age', 'Fare', 'FamilySize']] = sc.fit_transform(input_data[['Age', 'Fare', 'FamilySize']])  
     
    # Return output
    return input_data

# Import training data
#filepath = "C:/Users/Giloniss/Documents/Machine Learning/titanic/train.csv"
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
                         hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    
    return y_pred 

# Create simple models 
def females_survive(X_train, y_train, X_test):
    y_pred = 1 - X_test[:,3]
    return y_pred

def percent_survived(X_train, y_train, X_test):
    numel = len(X_test[:,0])
    num_survived = sum(X_test[:,0])
    prob_survival = num_survived/numel
    y_pred = rand(numel, 1, density=prob_survival, format='csr')
    y_pred.data[:] = 1
    y_pred = y_pred.toarray()
    return y_pred

classifiers = {"Proportion Survive": percent_survived,
               "Women Survive" : females_survive,
               "Logistic Regression" : default_logistic, 
               "Guassian Naive Bayes" : default_gauss_nb, 
               "Multinominal Naive Bayes" : default_multinom_nb,
               "Bernoulli Naive Bayes" : default_bernoulli_nb,
               "RBF SVM" : default_rbf_svm,
               "Linear SVM" : default_linear_svm,
               "Polynomial SVM" : default_poly_svm,
               "Random Forest" : default_rf,
               "Extra Tree" : default_extra_tree,
               "Neural Network" : default_nn
              }

# Functions to test accuracy
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

model_performance = pd.DataFrame(columns=['Model', 'Accuracy', 'Brier score', 'Area under ROC curve'])

for key, value in classifiers.items():
    y_pred = value(X_train, y_train, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    model_performance = model_performance.append({'Model': key, 'Accuracy': accuracy, 'Brier score': brier, 'Area under ROC curve': roc_auc}, ignore_index=True)
    # #Evaluate model
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    
# Random Forest algorithm has the highest accuracy, lowest Brier score and the highest area under the ROC curve
# This indicates that it is accurately indentifying survivors without false-positives and with appropriate probabilities.
    
# Re-fit the Random Forest to all of the training data
X_train = training_data_processed.iloc[:,1:].values
y_train = training_data_processed.iloc[:,0].values
clf = RandomForestClassifier() 
clf.fit(X_train, y_train)

# Import submission data
filepath = "C:/Users/samue/Dropbox/Machine Learning/titanic/test.csv"
submission_data = pd.read_csv(filepath)
submission_data_processed = pre_process(submission_data.copy())

# Predicting the test set results
X_sub = submission_data_processed.iloc[:,:].values
y_pred = clf.predict(X_sub)
y_pred = pd.DataFrame(y_pred, columns = ['Survived'])
submission_data = pd.concat([y_pred, submission_data], axis=1, sort=False)
    
# Submission
submission = submission_data[['PassengerId', 'Survived']]
submission.to_csv("C:/Users/samue/Dropbox/Machine Learning/titanic/random_forest_submission.csv", index=0)


