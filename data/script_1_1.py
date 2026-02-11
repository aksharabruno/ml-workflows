from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

def run_classification_example():
    # generate sample data
    print("Generating sample data...")
    x = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # train model
    print("Training a random forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train_scaled, y_train)

    # evaluate
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    run_classification_example()