import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


def prepare_data(csv_path):
    # Veriyi oku
    raw_data = pd.read_csv(csv_path)

    # Kullanılacak kolonları seç
    cols = ['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']
    data = raw_data[cols].copy()

    # Zaman bilgisini birleştir
    data['Timestamp'] = pd.to_datetime(
        data['Date'] + ' ' + data['Time'],
        errors='coerce'
    )

    # Geçersiz zamanları temizle
    data = data.dropna(subset=['Timestamp'])

    # Unix time hesapla
    data['Unix_Time'] = (
        data['Timestamp'] - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta(seconds=1)

    # --- EDA ---
    Path("outputs").mkdir(exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(data['Magnitude'], bins=30, kde=True)
    plt.title("Magnitude Distribution")

    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x='Magnitude',
        y='Depth',
        data=data,
        alpha=0.2
    )
    plt.title("Magnitude vs Depth")

    plt.tight_layout()
    plt.savefig("outputs/eda_plots.png")
    plt.close()

    # Model girdileri ve hedefler
    features = data[['Unix_Time', 'Latitude', 'Longitude']]
    targets = data[['Magnitude', 'Depth']]

    # Train / Test bölme
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42
    )

    # Ölçekleme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    prepare_data("data/database.csv")
