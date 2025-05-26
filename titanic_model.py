import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
sns.set(style="whitegrid")

# Load dataset
df = sns.load_dataset('titanic')

# Drop irrelevant columns and rows with missing data
df = df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'], axis=1)
df = df.dropna()

# Convert categorical columns to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features and target
X = df[['sex', 'age', 'fare', 'embarked', 'sibsp', 'parch']]
y = df['survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Example prediction
sample = [[1, 25, 70, 1, 0, 0]]
print("Prediction for sample passenger:", model.predict(sample))






#survival count plot visualization using seaborn and matplotlib

sns.countplot(x='survived', data=df)
plt.title('Survival Count')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()


#survival by sex visualization using seaborn and matplotlib

sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival by Sex')
plt.legend(labels=['Not Survived', 'Survived'])
plt.show()



#Age Distribution (Histogram) visualization
sns.histplot(data=df, x='age', kde=True, bins=30)
plt.title('Age Distribution')
plt.show()


# Box Plot of Fare by Survival

sns.boxplot(x='survived', y='fare', data=df)
plt.title('Fare Distribution by Survival')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()


#Heatmap of Correlation


corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()







