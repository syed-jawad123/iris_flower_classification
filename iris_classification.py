# Step 1: Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Step 3: Show dataset info neatly
print("========== IRIS DATASET PREVIEW ==========")
print(df.head(), "\n")
print("Dataset Shape:", df.shape)
print("Species Mapping:", dict(zip(range(3), iris.target_names)), "\n")

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("========== TRAIN/TEST SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0], "\n")

# Step 5: Train Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("========== MODEL PERFORMANCE ==========")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 8: Example Prediction
print("========== EXAMPLE PREDICTION ==========")
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Change values to test
prediction = model.predict(new_flower)
print("Input features:", new_flower[0])
print("Predicted Flower Type:", iris.target_names[prediction][0])