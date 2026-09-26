import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# =====> QCIRCUIT TRAINER <=====
def train_qiskit_classifier(
    X,
    y,
    test_size=0.2,
    random_state=42,
    feature_reps=2,
    ansatz_reps=2,
    maxiter=100,
):
    """
    Train a Qiskit variational quantum classifier on a six-feature dataset.

    Parameters
    ----------
    X : array-like, shape (n_samples, 6)
        Six-feature input dataset.

    y : array-like, shape (n_samples,)
        Binary labels, for example [0, 1].

    test_size : float
        Fraction of data used for testing.

    random_state : int
        Random seed for reproducibility.

    feature_reps : int
        Number of repetitions in the ZZ feature map.

    ansatz_reps : int
        Number of repetitions in the trainable quantum circuit.

    maxiter : int
        Maximum number of COBYLA optimization iterations.

    Returns
    -------
    result : dict
        Trained model, scaler, test data, predictions, and accuracy.
    """

    # Convert inputs to NumPy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Validate the feature count
    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError(
            f"X must have shape (n_samples, 6), but received {X.shape}"
        )

    # Validate binary labels
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError(
            f"This function requires exactly two classes, "
            f"but found {unique_labels}"
        )

    # Convert arbitrary binary labels to 0 and 1
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_int[label] for label in y])

    # Split before fitting the scaler to avoid test-set leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    # Standardize features, then map them to a useful rotation range.
    # ZZFeatureMap uses the values as rotation angles.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.clip(X_train, -1.0, 1.0) * np.pi
    X_test = np.clip(X_test, -1.0, 1.0) * np.pi

    # Reproducibility for Qiskit Machine Learning
    algorithm_globals.random_seed = random_state

    # Six features -> six qubits
    feature_map = ZZFeatureMap(
        feature_dimension=6,
        reps=feature_reps,
        entanglement="linear",
    )

    ansatz = RealAmplitudes(
        num_qubits=6,
        reps=ansatz_reps,
        entanglement="linear",
    )

    optimizer = COBYLA(maxiter=maxiter)

    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )

    # Train
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.3f}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[str(label) for label in unique_labels],
        )
    )

    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "label_mapping": label_to_int,
    }

# =====> QCIRCUIT PREDICTOR <=====

def predict_qiskit_classifier(model, scaler, X_new):
    """
    Predict labels for new samples with six features.
    """

    X_new = np.asarray(X_new, dtype=float)

    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    if X_new.shape[1] != 6:
        raise ValueError(
            f"X_new must have shape (n_samples, 6), but received {X_new.shape}"
        )

    X_new = scaler.transform(X_new)
    X_new = np.clip(X_new, -1.0, 1.0) * np.pi

    return model.predict(X_new)




# =====> MAIN <=====

def main():
    # X should have shape (number_of_rows, 6)
    # y should contain two classes, such as 0 and 1

    result = train_qiskit_classifier(X, y)

    model = result["model"]
    scaler = result["scaler"]

    new_predictions = predict_qiskit_classifier(
        model,
        scaler,
        X_new,
    )

    print(new_predictions)






















if __name__ == "__main__":
    main()