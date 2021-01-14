import pandas as ps
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
cols = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shuck_weight',
        'viscera_weight', 'shell_weight', 'rings']

# read data and set column names
abalone = ps.read_csv(data, names=cols)
abalone.head()
abalone.describe()
# one hot encode the sex column
encode = ps.get_dummies(abalone['sex'])
encoded_abalone = ps.concat([abalone, encode], axis=1)
# drop the now redundant sex column
encoded_abalone = encoded_abalone.drop(['sex'], axis=1)
# rename the columns to be more readable
encoded_abalone = encoded_abalone.rename(columns={'M': 'male', 'F': 'female', 'I': 'infant'})

# encode the rings data so that 1 represents if the rings are above the average and 0 if below
encoded_abalone.loc[:, 'above_avg_age'] = (encoded_abalone.loc[:, 'rings'] >= 10).astype(int)
# define the features and the targets for classification
X = encoded_abalone.drop(['rings', 'above_avg_age'], axis=1)
Y = encoded_abalone['above_avg_age']

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# take a best guess at the hyper parameters to use
cost = .9  # penalty parameter of the error term
gamma = 5  # defines the influence of input vectors on the margins

# test a LinearSVC with the initial hyper parameters
svc1 = svm.LinearSVC(C=cost).fit(X_train, y_train)
svc1.predict(X_test)
print("LinearSVC")
print(classification_report(svc1.predict(X_test), y_test))

# test linear, rbf and poly kernels
for k in ('rbf', 'poly'):
    svc = svm.SVC(gamma=gamma, kernel=k, C=cost).fit(X_train, y_train)
    svc.predict(X_test)
    print(k)
    print(classification_report(svc.predict(X_test), y_test))