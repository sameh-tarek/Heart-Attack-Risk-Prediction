import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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
data.dropna(inplace=True)
print(data)
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Diet'] = le.fit_transform(data['Diet'])
data['Blood Pressure'] = data['Blood Pressure'].apply(lambda x: float(x.split('/')[0]) if '/' in str(x) else float(x))
print(data['Diet'])
print(data['Sex'])
print(data['Blood Pressure'])


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

# decision tree
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
dtree = DecisionTreeClassifier(criterion='entropy') # gini or entropy
dtree.fit(x_train, y_train)
pred = dtree.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# decision tree Visualization
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, feature_names=x.columns, class_names=['0', '1', '2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('images/decisionTree/tree.png')
Image(graph.create_png())
