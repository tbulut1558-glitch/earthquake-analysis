import pandas as pd
import joblib
from tensorflow.keras.models import load_model


def predict_earthquake():
    try:
        model = load_model("models/earthquake_model.h5", compile=False)
        scaler = joblib.load("models/scaler.pkl")

        print("\nEarthquake Prediction Tool")

        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))
        date_input = input("Date (YYYY-MM-DD): ")

        timestamp = (
            pd.to_datetime(date_input) - pd.Timestamp("1970-01-01")
        ) // pd.Timedelta(seconds=1)

        input_df = pd.DataFrame(
            [[timestamp, latitude, longitude]],
            columns=["Unix_Time", "Latitude", "Longitude"]
        )

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input, verbose=0)

        print("\nPrediction Result")
        print(f"Estimated Magnitude : {prediction[0][0]:.2f}")
        print(f"Estimated Depth     : {prediction[0][1]:.2f} km")

    except Exception as err:
        print("Prediction failed:", err)


if __name__ == "__main__":
    predict_earthquake()
