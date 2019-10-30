
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def clf(X_train, y_train):
    clf = DecisionTreeClassifier(max_depth=3, random_state=123)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    y_pred_proba = clf.predict_proba(X_train)
    return clf, y_pred, y_pred_proba