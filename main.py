import pandas as ps
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
cols = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shuck_weight',
        'viscera_weight', 'shell_weight', 'rings']

data_forests = 'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'

# read data and set column names
abalone = ps.read_csv(data, names=cols)
forests = ps.read_csv(data_forests, header = 0)
abalone.head()
abalone.describe()
forests.head()
forests.describe()
# one hot encode the sex column
encode = ps.get_dummies(abalone['sex'])
encoded_abalone = ps.concat([abalone, encode], axis=1)
# one hot encode the month and day column
encode_forests_1 = ps.get_dummies(forests['month'])
encode_forests_2 = ps.get_dummies(forests['day'])
encoded_forests = ps.concat([forests, encode_forests_1], axis=1)
encoded_forests = ps.concat([encoded_forests, encode_forests_2], axis=1)
# drop the now redundant sex column
encoded_abalone = encoded_abalone.drop(['sex'], axis=1)
# drop the now redundant month adn day column
encoded_forests= encoded_forests.drop(['month'], axis=1)
encoded_forests= encoded_forests.drop(['day'], axis=1)
# rename the columns to be more readable
encoded_abalone = encoded_abalone.rename(columns={'M': 'male', 'F': 'female', 'I': 'infant'})
# encode the rings data so that 1 represents if the rings are above the average and 0 if below
encoded_abalone.loc[:, 'above_avg_age'] = (encoded_abalone.loc[:, 'rings'] >= 10).astype(int)
# encode the rain data so that 0 represents if there is no rain
encoded_forests.loc[:, 'rain'] = (encoded_forests.loc[:, 'rain'] > 0).astype(int)
# define the features and the targets for classification
X = encoded_abalone.drop(['rings', 'above_avg_age'], axis=1)
Y = encoded_abalone['above_avg_age']
X_2 = encoded_forests .drop(['wind', 'rain'], axis=1)
Y_2 = encoded_forests ['rain']

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, Y_2, test_size=0.5, random_state=0)

# take a best guess at the hyper parameters to use
cost = .9  # penalty parameter of the error term
gamma = 5  # defines the influence of input vectors on the margins

# test a LinearSVC with the initial hyper parameters
svc1 = svm.LinearSVC(C=cost).fit(X_train, y_train)
svc1.predict(X_test)
svc1_2 = svm.LinearSVC(C=cost).fit(X_2_train, y_2_train)
svc1_2.predict(X_2_test)


print("LinearSVC")
print(classification_report(svc1.predict(X_test), y_test))

# test linear, rbf and poly kernels
for k in ('rbf', 'poly'):
    svc = svm.SVC(gamma=gamma, kernel=k, C=cost).fit(X_train, y_train)
    svc_2 = svm.SVC(gamma=gamma, kernel=k, C=cost).fit(X_2_train, y_2_train)
    svc.predict(X_test)
    svc_2.predict(X_2_test)
    print(k)
    print(classification_report(svc.predict(X_test), y_test))
    print(classification_report(svc_2.predict(X_2_test), y_2_test))