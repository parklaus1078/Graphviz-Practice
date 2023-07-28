from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import graphviz as gv
import warnings
warnings.filterwarnings("ignore")

# Create DecisionTree Classifier
dt_clf = DecisionTreeClassifier(random_state=156)

# load iris data, and create train and test subsets
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=11)

# Execute DecisionTreeClassifier Training
dt_clf.fit(X_train, y_train)

export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names, feature_names=iris_data.feature_names, impurity=True, filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
gv.Source(dot_graph).view()

