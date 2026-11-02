from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout

from pipeline import (
    FEATURES,
    MODEL_PATH,
    TIME_STEPS,
    load_data,
    prepare_training_data,
    save_scaler,
)


def build_model() -> Sequential:
    model = Sequential(
        [
            Input(shape=(TIME_STEPS, len(FEATURES))),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main() -> None:
    data = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(data)
    model = build_model()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    print(f"Training LSTM with {len(X_train)} train and {len(X_test)} test sequences...")
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    save_scaler(scaler)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Scaled test loss: {test_loss:.6f}; scaled test MAE: {test_mae:.6f}")


if __name__ == "__main__":
    main()
