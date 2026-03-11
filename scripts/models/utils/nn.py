"""Keras MLP model definition, Optuna tuning, and training utilities."""

import numpy as np
import tensorflow as tf


def set_seed(seed=23):
    """Set random seeds for reproducibility across numpy, Python, and TensorFlow.

    Args:
        seed: Integer seed value (default 23).
    """
    import random

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


def _build_model(arch_params, input_shape):
    """Build a Keras Sequential MLP from a pre-sampled architecture parameter dict.

    Separates model construction from Optuna trial sampling so that the same
    architecture can be rebuilt with fresh weights for each CV fold without
    triggering additional trial.suggest calls.

    Args:
        arch_params: Dict with keys num_layers, units_i, activation_i,
            L1_i, dropout_i as sampled once per trial.
        input_shape: Number of input features.

    Returns:
        Uncompiled Keras Sequential model.
    """
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    set_seed()
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    for i in range(arch_params["num_layers"]):
        model.add(
            layers.Dense(
                arch_params[f"units_{i}"],
                activation=arch_params[f"activation_{i}"],
                kernel_regularizer=regularizers.L1(arch_params[f"L1_{i}"]),
            )
        )
        model.add(layers.Dropout(arch_params[f"dropout_{i}"]))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def objective(trial, data, r, folds, drop_cols=None):
    """Optuna objective function for Keras MLP hyperparameter tuning using CV.

    Samples all hyperparameters once per trial, then trains a fresh model
    with those parameters on each walk-forward fold, returning the mean
    validation loss across folds.

    Args:
        trial: Optuna trial object.
        data: Full modeling DataFrame.
        r: Tournament round number (used for feature construction).
        folds: List of fold dicts from make_folds().
        drop_cols: Optional list of feature names to exclude.

    Returns:
        Mean validation loss across all CV folds.
    """
    import gc

    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.optimizers import Adam, RMSprop

    from models.utils.DataProcessing import apply_smote, create_fold_splits, get_class_weights

    # Sample all hyperparameters once per trial
    num_layers = trial.suggest_int("num_layers", 1, 2)
    arch_params = {"num_layers": num_layers}
    for i in range(num_layers):
        arch_params[f"units_{i}"] = trial.suggest_int(f"units_{i}", 32, 256, step=32)
        arch_params[f"activation_{i}"] = trial.suggest_categorical(
            f"activation_{i}", ["relu", "tanh"]
        )
        arch_params[f"L1_{i}"] = trial.suggest_float(f"L1_{i}", 1e-6, 1e-2, log=True)
        arch_params[f"dropout_{i}"] = trial.suggest_float(f"dropout_{i}", 0.0, 0.7)

    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    optimizer_dict = {"adam": Adam, "rmsprop": RMSprop}

    fold_losses = []
    for fold_idx, fold in enumerate(folds):
        X_train, X_val, y_train, y_val = create_fold_splits(data, r, fold, drop_cols=drop_cols)
        X_train, y_train = apply_smote(X_train, y_train)
        class_weights = get_class_weights(y_train)
        # Convert per-sample array to {class: weight} dict for Keras
        unique_classes = np.unique(y_train)
        class_weight_dict = {int(c): float(class_weights[y_train == c][0]) for c in unique_classes}

        model = _build_model(arch_params, X_train.shape[1])
        optimizer = optimizer_dict[optimizer_name](learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["Precision"])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, ClearMemory()],
            verbose=0,
        )
        fold_loss = min(history.history["val_loss"])
        fold_losses.append(fold_loss)
        print(
            f"      Fold {fold_idx + 1}/{len(folds)}: val_loss={fold_loss:.4f} "
            f"(epochs={len(history.history['val_loss'])})"
        )

        K.clear_session()
        gc.collect()
        del model, history

    mean_loss = sum(fold_losses) / len(fold_losses)
    print(f"      Mean loss: {mean_loss:.4f}")
    return mean_loss


def tune_nn(data, r, folds, n_trials=75, drop_cols=None):
    """Tune Keras MLP hyperparameters using Optuna with walk-forward CV.

    Evaluates each trial across all CV folds and returns the hyperparameters
    that minimise mean validation loss, giving stable estimates that are not
    dependent on any single val window.

    Args:
        data: Full modeling DataFrame.
        r: Tournament round number used for feature construction.
        folds: List of fold dicts from make_folds().
        n_trials: Number of Optuna trials (default 75).
        drop_cols: Optional list of feature column names to exclude before
            scaling.

    Returns:
        Best hyperparameter dict from the Optuna study.
    """
    import os

    import optuna
    from optuna.visualization import plot_optimization_history

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    study = optuna.create_study(
        study_name=f"nn_round_{r}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=23),
    )

    def _trial_callback(study, trial):
        print(
            f"    Trial {trial.number + 1}/{n_trials}: loss={trial.value:.4f} "
            f"(best={study.best_value:.4f})"
        )

    study.optimize(
        lambda trial: objective(trial, data, r, folds, drop_cols=drop_cols),
        n_trials=n_trials,
        gc_after_trial=True,
        callbacks=[_trial_callback],
    )

    fig = plot_optimization_history(study)
    path = os.path.join(os.path.abspath(os.getcwd()), f"results/models/nn/round_{r}.png")
    fig.write_image(path)

    return study.best_params


def tuned_nn(params, X_train, y_train, X_val=None, y_val=None, class_weight=None):
    """Train a Keras MLP with pre-tuned hyperparameters.

    Args:
        params: Hyperparameter dict from a completed Optuna study.
        X_train: Training feature array.
        y_train: Training labels.
        X_val: Optional validation feature array for early stopping.
        y_val: Optional validation labels for early stopping.
        class_weight: Optional dict mapping class indices to weights, e.g.
            {0: 1.0, 1: 2.5}. Passed directly to model.fit() so the loss
            is scaled per class during training.

    Returns:
        Trained Keras Sequential model.
    """
    import os

    from tensorflow import keras
    from tensorflow.keras.optimizers import Adam, RMSprop

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    model = _build_model(params, X_train.shape[1])
    optimizer_dict = {"adam": Adam, "rmsprop": RMSprop}
    model.compile(
        optimizer=optimizer_dict[params["optimizer"]](learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["Precision"],
    )

    if (X_val is not None) and (y_val is not None):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=params["batch_size"],
            class_weight=class_weight,
            callbacks=[early_stopping, ClearMemory()],
            verbose=0,
        )
    else:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="loss", patience=30, restore_best_weights=True
        )
        model.fit(
            X_train,
            y_train,
            epochs=200,
            batch_size=params["batch_size"],
            class_weight=class_weight,
            callbacks=[early_stopping, ClearMemory()],
            verbose=0,
        )

    return model
