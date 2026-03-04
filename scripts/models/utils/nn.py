"""Keras MLP model definition, Optuna tuning, and training utilities."""

import tensorflow as tf


def set_seed(seed=23):
    """Set random seeds for reproducibility across numpy, Python, and TensorFlow.

    Args:
        seed: Integer seed value (default 23).
    """
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


class ClearMemory(tf.keras.callbacks.Callback):
    """Keras callback to run garbage collection after each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        """Trigger garbage collection to free memory after each epoch.

        Args:
            epoch: Current epoch index.
            logs: Optional dict of metric values.
        """
        import gc

        gc.collect()


def create_model(trial, input_shape):
    """Build a Keras Sequential MLP with architecture defined by Optuna trial.

    Args:
        trial: Optuna trial object used to suggest hyperparameters.
        input_shape: Number of input features.

    Returns:
        Compiled Keras Sequential model.
    """
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    set_seed()

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    num_layers = trial.suggest_int("num_layers", 1, 3)
    for i in range(num_layers):
        num_units = trial.suggest_int(f"units_{i}", 64, 320, step=32)
        activation = trial.suggest_categorical(f"activation_{i}", ["relu", "tanh"])
        model.add(
            layers.Dense(
                num_units,
                activation=activation,
                kernel_regularizer=regularizers.L1(trial.suggest_float(f"L1_{i}", 1e-9, 1e-3)),
            )
        )
        if trial.suggest_categorical(f"batch_norm_{i}", [True, False]):
            model.add(layers.BatchNormalization())
        dropout_rate = trial.suggest_float(f"dropout_{i}", 0.0, 1)
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def objective(trial, X_train, X_val, y_train, y_val):
    """Optuna objective function for Keras MLP hyperparameter tuning.

    Args:
        trial: Optuna trial object.
        X_train: Training feature array.
        X_val: Validation feature array.
        y_train: Training labels.
        y_val: Validation labels.

    Returns:
        Minimum validation loss achieved during training.
    """
    import gc

    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.optimizers.legacy import SGD, Adam, RMSprop

    model = create_model(trial, X_train.shape[1])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer_dict = {"adam": Adam, "rmsprop": RMSprop, "sgd": SGD}
    optimizer = optimizer_dict[optimizer_name](learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["Precision"])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        callbacks=[early_stopping, ClearMemory()],
        verbose=0,
    )
    loss = min(history.history["val_loss"])

    K.clear_session()
    gc.collect()
    del model, history

    return loss


def tune_nn(data, r, split_dict, n_trials=300):
    """Tune Keras MLP hyperparameters using Optuna for a given round.

    Splits data first, fits the scaler on the training fold only, then
    applies SMOTE resampling to the training fold to prevent data leakage.

    Args:
        data: Full modeling DataFrame.
        r: Tournament round number.
        split_dict: Dict mapping round number to train/val split ratio.
        n_trials: Number of Optuna trials (default 300).

    Returns:
        Best hyperparameter dict from the Optuna study.
    """
    import os

    import optuna
    from optuna.visualization import plot_optimization_history

    from models.utils.DataProcessing import apply_smote, create_splits

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    X_raw, y_raw = create_splits(data, r)
    split_idx = int(split_dict[r] * len(X_raw))
    X_train, X_val, y_train, y_val, _ = create_splits(data, r, split_idx=split_idx)
    X_train, y_train = apply_smote(X_train, y_train)

    study = optuna.create_study(
        study_name=f"nn_round_{r}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=23),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials,
        gc_after_trial=True,
    )

    fig = plot_optimization_history(study)
    path = os.path.join(os.path.abspath(os.getcwd()), f"results/models/nn/round_{r}.png")
    fig.write_image(path)

    return study.best_params


def tuned_nn(params, X_train, y_train, X_val=None, y_val=None):
    """Train a Keras MLP with pre-tuned hyperparameters.

    Args:
        params: Hyperparameter dict from a completed Optuna study.
        X_train: Training feature array.
        y_train: Training labels.
        X_val: Optional validation feature array for early stopping.
        y_val: Optional validation labels for early stopping.

    Returns:
        Trained Keras Sequential model.
    """
    import os

    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.optimizers.legacy import SGD, Adam, RMSprop

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    set_seed()

    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    num_layers = params["num_layers"]
    for i in range(num_layers):
        model.add(
            layers.Dense(
                params[f"units_{i}"],
                activation=params[f"activation_{i}"],
                kernel_regularizer=regularizers.L1(params[f"L1_{i}"]),
            )
        )
        if params[f"batch_norm_{i}"]:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(params[f"dropout_{i}"]))

    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer_dict = {"adam": Adam, "rmsprop": RMSprop, "sgd": SGD}
    model.compile(
        optimizer=optimizer_dict[params["optimizer"]](learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["Precision"],
    )

    if (X_val is not None) and (y_val is not None):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=params["batch_size"],
            callbacks=[early_stopping, ClearMemory()],
            verbose=0,
        )
    else:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=params["batch_size"],
            callbacks=[early_stopping, ClearMemory()],
            verbose=0,
        )

    return model
