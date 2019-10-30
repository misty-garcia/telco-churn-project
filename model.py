from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#random state set for 123 for all models

# logistic regression model
def logit(X_train, y_train):
    logit = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga')
    logit.fit(X_train, y_train)

    y_pred = logit.predict(X_train)
    y_pred_proba = logit.predict_proba(X_train)
    return logit, y_pred, y_pred_proba    

# decision tree model
def clf(X_train, y_train):
    clf = DecisionTreeClassifier(max_depth=3, random_state=123)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    y_pred_proba = clf.predict_proba(X_train)
    return clf, y_pred, y_pred_proba   

# random forest model
def rf(X_train, y_train):
    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', min_samples_leaf=3, n_estimators=100, max_depth=3,random_state=123)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_train)
    y_pred_proba = rf.predict_proba(X_train)
    return rf, y_pred, y_pred_proba   

# k-nearest neighbor
def knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=4, weights='uniform')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_train)
    y_pred_proba = knn.predict_proba(X_train)
    return knn, y_pred, y_pred_proba