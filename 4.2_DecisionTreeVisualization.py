from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dt_clf = DecisionTreeClassifier(random_state=156)
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=11)
dt_clf.fit(X_train, y_train)

feat_impt = dt_clf.feature_importances_
print("Feature importance :\n{0}".format(np.round(feat_impt, 3)))
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print("{0} : {1:.3f}".format(name, value))
sns.barplot(x=feat_impt, y=iris_data.feature_names)
plt.show()

export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names, feature_names=iris_data.feature_names, impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph).view()

plt.title("3 Class values with 2 Features Sample data Creation")
X_features, y_labels = make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes=3, n_clusters_per_class=1, random_state=0)
plt.scatter(X_features[:, 0], X_features[:, 1], marker="o", c=y_labels, s=25, cmap="rainbow", edgecolors="k")
plt.show()

def visualize_boundary(model, X, y):
    fig,ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap="rainbow", edgecolor="k", clim=(y.min(), y.max()), zorder=3)
    ax.axis("tight")
    ax.axis("off")
    xlim_start, xlim_end = ax.get_xlim()
    ylim_start, ylim_end = ax.get_ylim()

    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(xlim_start, xlim_end, num=200), np.linspace(ylim_start, ylim_end, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes+1) - .5, cmap="rainbow", clim=(y.min(), y.max()), zorder=1)
    plt.show()


dt_clf = DecisionTreeClassifier(random_state=156).fit(X_features, y_labels)
visualize_boundary(dt_clf, X_features, y_labels)

dt_clf = DecisionTreeClassifier(random_state=156, min_samples_leaf=6).fit(X_features, y_labels)
visualize_boundary(dt_clf, X_features, y_labels)