from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

def run_classification_example():
    # generate sample data
    print("Generating synthetic patient data...")
    patient_features = np.random.randn(1000, 20)
    disease_labels = np.random.randint(0, 2, 1000)

    # split data with custom names
    training_cohort, validation_cohort, training_outcomes, validation_outcomes = train_test_split(
        patient_features, disease_labels, test_size=0.2, random_state=42)
    
    # normalize
    feature_normalizer = StandardScaler()
    normalized_training_cohort = feature_normalizer.fit_transform(training_cohort)
    normalized_validation_cohort = feature_normalizer.transform(validation_cohort)

    print("Training a random forest model...")
    disease_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    disease_classifier.fit(normalized_training_cohort, training_outcomes)

    #predict
    predicted_outcomes = disease_classifier.predict(normalized_validation_cohort)
    classfication_performance = accuracy_score(validation_outcomes, predicted_outcomes)

    print(f"Performance: {classfication_performance:.2f}")

if __name__ == "__main__":
    run_classification_example()