import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from six import StringIO
from IPython.display import Image
import pydotplus

# read data
data = pd.read_csv('heart_attack_prediction_dataset.csv')
data = data.loc[:, ('Age', 'Sex', 'Cholesterol', 'Blood Pressure', 'Diabetes', 'Smoking', 'Obesity',
                    'Exercise Hours Per Week', 'Diet', 'Previous Heart Problems', 'Medication Use',
                    'Stress Level', 'BMI', 'Physical Activity Days Per Week', 'Heart Attack Risk')]
numeric_columns = data.select_dtypes(include=['number']).columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
print(numeric_columns)
print(non_numeric_columns)
print('\n\n')
print(data)

# Visualize Numeric Columns
for col in numeric_columns:
    plt.figure(figsize=(10, 6))

    # Line Plot
    plt.subplot(2, 2, 1)
    sns.lineplot(x=data.index, y=data[col])
    plt.title(f'Line Plot of {col}')

    # Histogram plot
    plt.subplot(2, 2, 2)
    sns.histplot(data[col], bins=20)
    plt.title(f'Histogram of {col}')

    # Density plot
    plt.subplot(2, 2, 3)
    sns.kdeplot(data[col], fill=True)
    plt.title(f'Density Plot of {col}')

    # Vertical Bar plot
    plt.subplot(2, 2, 4)
    sns.barplot(x=data.index, y=data[col])
    plt.title(f'Vertical Bar of {col}')

    plt.tight_layout()
    #plt.savefig(f'images/{col}_plots.png')
    plt.show()


# preprocessing
print("\n************ preprocessing ************")
print("check missing values:\n", data.isnull().sum(), '\n')
# data.dropna(inplace=True)
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Diet'] = le.fit_transform(data['Diet'])
data['Blood Pressure'] = data['Blood Pressure'].apply(lambda x: float(x.split('/')[0]) if '/' in str(x) else float(x))
print("After preprocessing:\n")
print(data['Diet'], "\n")
print(data['Sex'], "\n")
print(data['Blood Pressure'], "\n")


# Visualize Numeric Columns again
for col in numeric_columns:
    plt.figure(figsize=(10, 6))

    # Line Plot
    plt.subplot(2, 2, 1)
    sns.lineplot(x=data.index, y=data[col])
    plt.title(f'Line Plot of {col}')

    # Histogram plot
    plt.subplot(2, 2, 2)
    sns.histplot(data[col], bins=20)
    plt.title(f'Histogram of {col}')

    # Density plot
    plt.subplot(2, 2, 3)
    sns.kdeplot(data[col], fill=True)
    plt.title(f'Density Plot of {col}')

    # Vertical Bar plot
    plt.subplot(2, 2, 4)
    sns.barplot(x=data.index, y=data[col])
    plt.title(f'Vertical Bar of {col}')

    plt.tight_layout()
    # plt.savefig(f'images/afterPreprocessing/{col}_plots.png')
    plt.show()

# features(x) and target(x)
x = data.drop('Heart Attack Risk', axis=1)
y = data['Heart Attack Risk']

# decision tree classification
print("\n**************** Decision tree classification ******************")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
dtree = DecisionTreeClassifier(criterion='entropy') # gini or entropy
dtree.fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)
accuracy = accuracy_score(y_test, dtree_pred)
print("Accuracy: ", accuracy)

# decision tree Visualization
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, feature_names=x.columns, class_names=['0', '1', '2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('images/decisionTree/tree.png')
Image(graph.create_png())

# KNN classification
print("\n************ KNN classification *************")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
knn_matrix = confusion_matrix(y_test, knn_pred)
print("confusion_matrix:\n", knn_matrix)

# accuracy score
acc = accuracy_score(y_test,  knn_pred)
print("accuracy: ", acc)
# precision
pre = precision_score(y_test,  knn_pred)
# recall
rec = recall_score(y_test,  knn_pred)
print("recall: ", rec)
# fi-measure
f1 = f1_score(y_test,  knn_pred)
print("f1-measure: ", f1)


# SVM classification
print("\n*********** SVM Classification *************")
SVM_Model = SVC(gamma='auto')
SVM_Model.fit(x_train, y_train)
svm_pred = SVM_Model.predict(x_test)
svm_matrix = confusion_matrix(y_test, svm_pred)
print("confusion_matrix:\n", svm_matrix)
# SVM accuracy
acc = accuracy_score(y_test, svm_pred)
print("SVM Accuracy: ", acc)

# ensemble model
print("\n*********** Ensemble *************")
ensemble_model = VotingClassifier(estimators=[
    ('DecisionTree', dtree),
    ('KNN', knn_model),
    ('SVM', SVM_Model)
], voting='hard')

ensemble_model.fit(x_train, y_train)
ensemble_pred = ensemble_model.predict(x_test)
ensemble_matrix = confusion_matrix(y_test, ensemble_pred)
print("Ensemble Confusion Matrix:\n", ensemble_matrix)

ensemble_acc = accuracy_score(y_test, ensemble_pred)
print("Ensemble Accuracy: ", ensemble_acc)


