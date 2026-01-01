import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from data_preprocessing import prepare_data
from pathlib import Path


def train_model():
    X_train, X_test, y_train, y_test = prepare_data("data/database.csv")

    model = Sequential()
    model.add(Input(shape=(3,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(2, activation="linear"))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    print("Model training started...")
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    Path("outputs").mkdir(exist_ok=True)

    # Eğitim kaybı grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Process")
    plt.legend()
    plt.savefig("outputs/training_history.png")
    plt.close()

    # Performans değerlendirme
    predictions = model.predict(X_test)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test.iloc[:, 0], predictions[:, 0], alpha=0.3)
    plt.plot(
        [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()],
        [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()],
        "r--"
    )
    plt.xlabel("Actual Magnitude")
    plt.ylabel("Predicted Magnitude")
    plt.title("Magnitude Comparison")

    plt.subplot(1, 2, 2)
    plt.scatter(y_test.iloc[:, 1], predictions[:, 1], alpha=0.3)
    plt.plot(
        [y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()],
        [y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()],
        "r--"
    )
    plt.xlabel("Actual Depth")
    plt.ylabel("Predicted Depth")
    plt.title("Depth Comparison")

    plt.tight_layout()
    plt.savefig("outputs/performance_comparison.png")
    plt.close()

    Path("models").mkdir(exist_ok=True)
    model.save("models/earthquake_model.h5")
    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()
