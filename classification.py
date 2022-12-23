from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn import preprocessing
import pandas as pd
import numpy as np

# Import dataset:
data = pd.read_csv('clustered_data.csv',index_col=False)

data['Policy'] = np.where(data['status_report'] == 'Policycreated', 1, 0)

data = pd.concat([data,data[data['Policy'] == 1].sample(frac=1),data[data['Policy'] == 1].sample(frac=1),data[data['Policy'] == 1].sample(frac=1)])
print('Policy Opted : ',len(data[data['Policy'] == 1]))
print('Policy Not Opted : ',len(data[data['Policy'] == 0]))
data = data.sample(frac=1)

x_columns = ['PROVINCES','URB','INCOME','SOCCL_A','SOCCL_B1','SOCCL_B2','SOCCL_C','SOCCL_D','EDU_HIGH','EDU_MID','EDU_LOW','DINK','OWN_HOUSE','AVG_HOUSE','RENT_PRICE','STAGE_OF_LIFE','SINGLE','FAM','FAM_WCHILD','SINGLES_YOUNG','SINGLES_MID','SINGLES_OLD','FAM_CHILD_Y','FAM_CHILD_O','FAM_WCHILD_Y','FAM_WCHILD_MED','FAM_WCHILD_OLD','CIT_HOUSEHOLD','LOAN','SAVINGS','SHOP_ONLINE','CAR']
y_columns = ['Policy','Cluster']



# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(data[x_columns], data[y_columns],random_state = 35, test_size=0.20, shuffle=True)
y_test = np.array(y_test)
# Standardize features by removing mean and scaling to unit variance:
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifiers = {'DecisionTreeClassifier':MultiOutputClassifier(DecisionTreeClassifier(random_state=35)),
               'RandomForestClassifier':MultiOutputClassifier(RandomForestClassifier(random_state=35)),
               'GradientBoostingClassifier':MultiOutputClassifier(GradientBoostingClassifier(random_state=35)),
               'KNeighborsClassifier':MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))}

for name,classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)


    print(name+ '\n')
    print('Policy')
    print(confusion_matrix(y_test[:, 0], y_pred[:, 0]))
    print(classification_report(y_test[:, 0], y_pred[:, 0]))
    print('Accuracy : ' + str(accuracy_score(y_test[:, 0], y_pred[:, 0])))
    print('\n')
    print('Cluster')
    print(confusion_matrix(y_test[:, 1], y_pred[:, 1]))
    print(classification_report(y_test[:, 1], y_pred[:, 1]))
    print('Accuracy : ' + str(accuracy_score(y_test[:, 1], y_pred[:, 1])))
    print('------------------------------------------\n')


classifier = MultiOutputClassifier(RandomForestClassifier(random_state=35))
classifier.fit(data[x_columns], data[y_columns])

tnb_policy_data = pd.read_excel('W27308.xlsx')
regional_data = pd.read_excel('W27309.xlsx')

test_data = tnb_policy_data.merge(regional_data, how="inner", on=['zipcode_link'])
le = preprocessing.LabelEncoder()

test_data['PROVINCES'] = le.fit_transform(test_data['PROVINCE'])
test_data.fillna(0,inplace = True)
prediction = classifier.predict(test_data[x_columns])

test_data['Policy'] = prediction[:, 0]
test_data['Cluster'] = prediction[:, 1]

test_data.to_csv('TnB_Policy_Dataset.csv')






