import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential

def create_model(X_train_scaled, y_train, X_val_scaled, y_val):
    model = Sequential([
        Dense(36, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(36, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(18, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(9, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    model.summary()
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=600,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    # Evaluate the model
    y_pred = (model.predict(X_val_scaled) > 0.5).astype(int).reshape(-1)
    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    return model
# The model is trained and evaluated, and the trained model is returned.
# The model architecture includes:
# - Input layer with 36 neurons and ReLU activation
# - Batch normalization and dropout layers to prevent overfitting
# - Hidden layers with 36, 18, and 9 neurons, each followed by batch normalization and dropout
# - Output layer with 1 neuron and sigmoid activation for binary classification
# The model is compiled with Adam optimizer and binary crossentropy loss function.
# Early stopping is used to prevent overfitting during training.
# The model is trained for 600 epochs with a batch size of 32.
# The model is evaluated on the validation set, and classification report and confusion matrix are printed.
# The trained model is returned for further use or predictions.