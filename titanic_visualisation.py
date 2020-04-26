import pandas as pd
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/samue/anaconda3/Library/bin/graphviz/'
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef

# Import training data
filepath = "C:/Users/samue/Dropbox/Machine Learning/titanic/train.csv"
training_data = pd.read_csv(filepath)

# Import submission data
filepath = "C:/Users/samue/Dropbox/Machine Learning/titanic/test.csv"
submission_data = pd.read_csv(filepath)

# Combine for imputing etc.
combined_data = [training_data.iloc[:,1:], submission_data]
combined_data = pd.concat(combined_data)

# Remove ticket number. N.B. This could potentially be used to identify co-travellers (look for proximate ticket no.s)
training_data.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)

# Missing cabin info 
training_data['Cabin'][training_data['Cabin'].isnull()] = "U"
training_data['Cabin'][training_data['Cabin'] == 'T'] = "U"

# # Extract a 'Deck' variable from 'Cabin'. Could also potentially take a room no. variable which might correspond to position fore/aft
training_data['Deck'] = training_data['Cabin'].astype(str).str[0]
training_data.drop(['Cabin'], axis=1, inplace=True)

# Extract new categorical 'stage of life' variable
training_data['Stage'] = ["Baby" if age < 1 \
                  else "Child" if age < 13 \
                  else "Teenager" if age < 18 \
                  else "Adult" if age < 50 \
                  else "Elderly" \
                  for age in training_data['Age']]
    
# Impute missing embarkation points (modal values) - only two missing from training data
training_data['Embarked'][training_data['Embarked'].isnull()] = "S"

# Aggregate Parch and SibSp into one variable
def family_size(family_data):
    family_data['FamilySize'] = family_data['Parch'] + family_data['SibSp'] + 1
    family_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)
    
    family_data['FamilyType'] = ['Lone' if family_size == 1 
                                else 'Small' if family_size < 5
                                else 'Large'\
                          for family_size in family_data['FamilySize']]
    return family_data
training_data = family_size(training_data)
combined_data = family_size(combined_data)
        

# Extract titles from names - use REGEXP to take everything after ', ' and before '.'
def extract_titles(title_data):
    title_data['Title'] = title_data['Name'].str.extract(pat = '((?<=,\s)[a-zA-Z\s]+(?=\.))')
    title_data['Title'][title_data['Title'].isin(["Mlle","Ms"])] = "Miss"
    title_data['Title'][title_data['Title'].isin(["Mme"])] = "Mrs"
    title_data['Title'][title_data['Title'].isin(["Rev","Dr","Major","Col","Capt"])] = "Officer"
    title_data['Title'][title_data['Title'].isin(["Don","Lady","Sir","the Countess","Jonkheer"])] = "Royalty"
    ix = title_data['Title'].isin(['Mr','Mrs', 'Master', 'Miss', 'Officer', 'Royalty'])
    title_data['Title'][~ix] = "Other"
    # Extract surname
    title_data['Surname'] = title_data['Name'].str.extract(pat = '([a-zA-Z\s]+(?=\,))')
    training_data['Surname'] = title_data['Name'].str.extract(pat = '([a-zA-Z\s]+(?=\,))')
    # Drop 'Name' field
    title_data.drop(['Name'], axis=1, inplace=True)
    return title_data
training_data = extract_titles(training_data)
combined_data = extract_titles(combined_data)

# Find % of passengers with the same surname that died.
surname_survived = training_data.groupby(['Surname']).agg('mean')['Survived']
surname_survived = surname_survived.reset_index()
def family_survived(row):
    ix = surname_survived['Surname'] == row['Surname']
    return surname_survived['Survived'][ix].values[0]
training_data['FamilySurvived'] = training_data.apply(lambda row: family_survived(row) 
                                                if (training_data['Surname']==row['Surname']).any() 
                                                else surname_survived['Survived'].mean(), axis=1)
training_data.drop(['Surname'], axis=1, inplace=True)

# Impute missing ages (median values)
ages = combined_data.groupby(['Sex', 'Pclass', 'Title']).agg('median')['Age']
ages = ages.reset_index()
def fill_ages(row):
    ix = (
        (ages['Sex'] == row['Sex']) &
        (ages['Pclass'] == row['Pclass']) &
        (ages['Title'] == row['Title'])
        )
    return ages['Age'][ix].values[0]
training_data['Age'] = training_data.apply(lambda row: fill_ages(row) if np.isnan(row['Age']) else row['Age'], axis=1)

# Impute missing fares (median values)
fares = combined_data.groupby(['Pclass', 'Embarked', 'FamilySize']).agg('median')['Fare']
fares = fares.reset_index()
def fill_fares(row):
    ix = (
        (fares['Pclass'] == row['Pclass']) &
        (fares['Embarked'] == row['Embarked']) &
        (fares['FamilySize'] == row['FamilySize'])
        )
    return fares['Fare'][ix].values[0]
training_data['Fare'] = training_data.apply(lambda row: fill_fares(row) if np.isnan(row['Fare']) else row['Fare'], axis=1)

# # Add log of fares
training_data['Log(Fare)'] = np.log(training_data['Fare'])
min_fare = training_data['Fare'][training_data['Fare'] != 0].min()
training_data['Log(Fare)'][training_data['Log(Fare)'] == -np.inf] = np.log(min_fare)
training_data.drop(['Fare'], axis=1, inplace=True)

# Following fields can be made categorical: Pclass, Sex, Embarked, Deck, Title
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first')
#for field in ['Pclass', 'Sex', 'Embarked', 'Deck', 'Title']:
categorical_fields = list(training_data.select_dtypes(include=['object']).columns)
categorical_fields.append('Pclass')
categorical_data = training_data[categorical_fields]
categorical_data = ohe.fit_transform(categorical_data).toarray() 
headers = ohe.get_feature_names(categorical_fields)
categorical_data = pd.DataFrame(categorical_data, columns = headers)
training_data.drop(categorical_fields, axis=1, inplace=True)
training_data = pd.concat([training_data, categorical_data], axis=1, sort=False)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
training_data[['Age', 'Log(Fare)', 'FamilySize']] = sc.fit_transform(training_data[['Age', 'Log(Fare)', 'FamilySize']])

# Drop features
# input_data.drop(['Embarked_Q','Deck_F','Deck_G','Title_Officer','Pclass_3'], axis=1, inplace=True)

    
# Convert categorical data types
le = LabelEncoder()
categorical_fields = list(training_data.select_dtypes(include=['object']).columns)
categorical_data = training_data[categorical_fields].apply(le.fit_transform)
training_categorical = training_data.drop(categorical_fields, axis=1)
training_categorical = pd.concat([training_categorical, categorical_data], axis=1, sort=False)

# Create dummy variables
ohe = OneHotEncoder(drop='first')
categorical_fields = list(training_data.select_dtypes(include=['object']).columns)
categorical_fields.append('Pclass')
categorical_data = training_data[categorical_fields]
categorical_data = ohe.fit_transform(categorical_data).toarray() 
headers = ohe.get_feature_names(categorical_fields)
categorical_data = pd.DataFrame(categorical_data, columns = headers)
training_dummy = training_data.drop(categorical_fields, axis=1, inplace=True)
training_dummy = pd.concat([training_dummy, categorical_data], axis=1, sort=False)

# Spearmans Rank correlation matrix
training_correlation = training_categorical.drop(['Embarked', 'Deck', 'Title', 'Stage', 'Died'], axis=1)
spearman_matrix = training_correlation.corr(method="spearman")
fig = plt.figure(figsize=(12, 10))
sns.heatmap(spearman_matrix, annot=True)

# Pearson's correlation
pearson_matrix = training_correlation.corr(method="pearson")
fig = plt.figure(figsize=(12, 10))
sns.heatmap(pearson_matrix, annot=True)

# Phi coefficient to measure association between binary variables
print(matthews_corrcoef(training_categorical['Sex'], training_categorical['Survived']))
print(matthews_corrcoef(training_categorical['IsAlone'], training_categorical['Survived']))

# Check proportion of each sex that survived.
training_data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True);

# Check proportion of each Pclass that survived.    
training_data.groupby('Pclass').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True);

# Age distribution for both genders
fig = plt.figure(figsize=(25, 7))
plt.hist(training_data['Age'][training_data['Sex']=='male'], bins=5, label = ['Male'])
plt.hist(training_data['Age'][training_data['Sex']=='female'], bins=5, label = 'Female')
plt.legend(loc="upper right")

# Age and title:
training_data.hist(column='Age', by='Title',bins=8, figsize=(14, 7))
plt.title('Age by title')

# Age and family size
training_data.hist(column='Age', by='FamilySize',bins=8, figsize=(14, 7))
plt.title('Age by family size')

# Effect of age on probability of survival:
fig = plt.figure(figsize=(25, 7))
rounded_ages = training_data[['Age','Survived']]
rounded_ages['Age'] = 5 * round(rounded_ages['Age']/5) # round to nearest 5 years
age_survival = rounded_ages.groupby('Age').agg('mean')['Survived']
age_survival = age_survival.reset_index()
age_survival.plot.scatter(x='Age', y='Survived', )
plt.ylabel('Survival rate')

# Female survivorship at different stages of life
females_stage = training_data[training_data['Sex']=='female'].groupby('Stage').agg('mean')[['Survived', 'Died']]
females_stage = females_stage.reindex(['Baby', 'Child', 'Teenager', 'Adult', 'Elderly'])
females_stage.plot(kind='bar', figsize=(25, 7), stacked=True);
plt.title('Females')

# Male survivorship at different stages of life
males_stage = training_data[training_data['Sex']=='male'].groupby('Stage').agg('mean')[['Survived', 'Died']]
males_stage = males_stage.reindex(['Baby', 'Child', 'Teenager', 'Adult', 'Elderly'])
males_stage.plot(kind='bar', figsize=(25, 7), stacked=True);
plt.title('Males')

# How do age and sex affect survival rate?
fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Sex', y='Age', 
               hue='Survived', data=training_data,
               scale='count',
               split=True,
               palette={0: "r", 1: "g"},
              );

# Check the distribution of Pclass at different ages
corr, _ = pearsonr(training_data['Pclass'], training_data['Age'])
pclass_age = training_data.groupby(['Stage','Pclass']).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack()
pclass_age = pclass_age.reindex(['Baby', 'Child', 'Teenager', 'Adult', 'Elderly'])
pclass_age.plot(kind='bar', figsize=(25, 7),stacked=True)
plt.ylabel('%')

# Check survival rate of each title
title_survival = training_data.groupby('Title').agg('mean')[['Survived', 'Died']]
title_survival = title_survival.reindex(['Mr', 'Mrs', 'Master', 'Miss', 'Exotic title'])
title_survival.plot(kind='bar', figsize=(25, 7), stacked=True);

# Fare paid
plt.hist(training_data['Fare'], bins=24, label = ['Fare'])
plt.title('Fares paid')

# Check effect of fare paid on survival chances
fig = plt.figure(figsize=(25, 7))
rounded_fares = training_data[['Fare','Survived']]
rounded_fares['Fare'] = 10 * round(rounded_fares['Fare']/10) # round to nearest 5£
fare_survival = rounded_fares.groupby('Fare').agg('mean')['Survived']
fare_survival = fare_survival.reset_index()
fare_survival.plot.scatter(x='Fare', y='Survived')
plt.ylabel('Survival rate')

# Take logs of fare paid
log_fares = training_data[['Fare','Survived']][training_data['Fare'] != 0]
log_fares['log_price'] = np.log(log_fares['Fare'])
fig = plt.figure(figsize=(25, 7))
#log_fares['log_price'] = log_fares['log_price'].round(1)
log_price_survival = log_fares.groupby('log_price').agg('mean')['Survived']
log_price_survival = log_price_survival.reset_index()
log_price_survival.plot.scatter(x='log_price', y='Survived')
plt.ylabel('Survival rate')
corr, _ = pearsonr(log_price_survival['log_price'], log_price_survival['Survived'])

# Test heuristics
ix = (training_data['Sex'] == 'male') & (training_data['Stage'] == 'Teenager') & (training_data['Pclass'] == 3)
poor_teenage_boys = training_data[ix]
ix = (training_data['Sex'] == 'female') & (training_data['Stage'] == 'Elderly') & (training_data['Pclass'] == 1)
rich_grandmother = training_data[ix]

# Decks by Pclass
decks_pclass = training_data.groupby(['Deck','Pclass']).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack()
decks_pclass = decks_pclass.reindex(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'U'])
decks_pclass.plot(kind='bar', figsize=(25, 7),stacked=True)
plt.ylabel('%')

# Survival by deck
decks_survival = training_data.groupby('Deck').agg('mean')[['Survived', 'Died']]
decks_survival.reindex(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'U'])
decks_survival.plot(kind='bar', figsize=(25, 7), stacked=True);

# Montecarlo simulation for survival by deck
p = []
for j in range(9999):
    count = 0
    for i in range(15):
        rnd = random.uniform(0, 1)
        if rnd<0.6:
            count = count+1           
    p.append(count)
    
# Check embarkation point
fig = plt.figure(figsize=(25, 7))
training_data.groupby('Embarked').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True);

# Pclass by embarkation point
embarked_pclass = training_data.groupby(['Embarked','Pclass']).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack()
embarked_pclass.plot(kind='bar', figsize=(25, 7),stacked=True)
plt.ylabel('%')

# SibSp
sibsp_survival = training_data.groupby('SibSp').agg('mean')['Survived']
sibsp_survival = sibsp_survival.reset_index()
sibsp_survival.plot.scatter(x='SibSp', y='Survived', )
plt.ylabel('Survival rate')
corr, _ = pearsonr(sibsp_survival['SibSp'], sibsp_survival['Survived'])

# Parch
parch_survival = training_data.groupby('Parch').agg('mean')['Survived']
parch_survival = parch_survival.reset_index()
parch_survival.plot.scatter(x='Parch', y='Survived', )
plt.ylabel('Survival rate')
corr, _ = pearsonr(parch_survival['Parch'], parch_survival['Survived'])

# Family size
family_survival = training_data.groupby('FamilySize').agg('mean')['Survived']
family_survival = family_survival.reset_index()
family_survival.plot.scatter(x='FamilySize', y='Survived', )
plt.ylabel('Survival rate')
corr, _ = pearsonr(family_survival['FamilySize'], family_survival['Survived'])

# Montecarlo simulation for % survived model
count = 0
num_sims = 100
for j in range(num_sims-1):  
    rndA = random.uniform(0, 1)
    if rndA <= 0.39:
        surv = 1
    else:
        surv = 0
    rndB = random.uniform(0, 1) 
    if rndB <= 0.39:
        guess = 1
    else:
        guess = 0        
    if surv == guess:
        count += 1
print(count/num_sims)