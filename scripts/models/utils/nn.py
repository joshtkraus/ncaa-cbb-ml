def set_seed(seed=23):
    import numpy as np
    import random
    import tensorflow as tf
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def create_model(trial, input_shape):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    set_seed()
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    # Tune the number of layers
    num_layers = trial.suggest_int("num_layers", 1, 3)
    for i in range(num_layers):
        num_units = trial.suggest_int(f"units_{i}", 64, 320, step=32)
        activation = trial.suggest_categorical(f"activation_{i}", ['relu', 
                                                                   'tanh'])
        model.add(layers.Dense(num_units,
                               activation=activation,
                               kernel_regularizer=regularizers.L1(trial.suggest_float(f"L1_{i}", 1e-9, 1e-3))
                                )
                )
        
        # Tune BatchNormalization
        if trial.suggest_categorical(f"batch_norm_{i}", [True, False]):
            model.add(layers.BatchNormalization())
        
        # Tune dropout rate
        dropout_rate = trial.suggest_float(f"dropout_{i}", 0.0, 1)
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation="sigmoid"))
    
    return model    

import tensorflow as tf
class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        import gc
        gc.collect()
def objective(trial, X_train, X_val, y_train, y_val):
    from tensorflow import keras
    from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD
    from tensorflow.keras import backend as K
    import gc

    # Create Model
    model = create_model(trial, X_train.shape[1])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    optimizer_dict = {"adam": Adam, "rmsprop": RMSprop, "sgd": SGD}
    optimizer = optimizer_dict[optimizer_name](learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["Precision"])

    # Early Stopping & Pruning
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Fit
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        callbacks=[early_stopping, ClearMemory()],
        verbose=0
    )
    loss = min(history.history['val_loss'])
    
    # Clear Memory
    K.clear_session()
    gc.collect()
    del model, history

    return loss

def tune_nn(data, r, split_dict, n_trials=300):
    import os
    import numpy as np
    import optuna
    from optuna.visualization import plot_optimization_history
    from models.utils.DataProcessing import create_splits
    import tensorflow as tf

    # Supress Logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Set Memory Allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Create Data
    X_SMTL, y_SMTL = create_splits(data, r, train=True)
    X, y = create_splits(data, r, train=False)

    # Data Splits
    split_idx = int(split_dict[r] * len(X))
    split_idx_SMTL = np.where((X_SMTL == X[split_idx]).all(axis=1))[0][0]
    X_train, X_val = X_SMTL[:split_idx_SMTL], X[split_idx:]
    y_train, y_val = y_SMTL[:split_idx_SMTL], y[split_idx:]

    # Tuning
    study = optuna.create_study(study_name=f"nn_round_{r}",
                                direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=23))
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), 
                   n_trials=n_trials,
                   gc_after_trial=True)
    
    # Save Plot
    fig = plot_optimization_history(study)
    path = os.path.join(os.path.abspath(os.getcwd()), f"results/models/nn/round_{r}.png")
    fig.write_image(path)
    
    return study.best_params

def tuned_nn(params, X_train, y_train, X_val=None, y_val=None):
    import os
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD

    # Logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Set Seed
    set_seed()
    
    # Create Model
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    
    # Number of layers
    num_layers = params['num_layers']
    for i in range(num_layers):
        num_units = params[f"units_{i}"]
        activation = params[f"activation_{i}"]

        model.add(layers.Dense(num_units,
                               activation=activation,
                               kernel_regularizer=regularizers.L1(params[f"L1_{i}"])
                                )
                )
        
        # BatchNormalization
        if params[f"batch_norm_{i}"]:
            model.add(layers.BatchNormalization())
        
        # Dropout Rate
        dropout_rate = params[f"dropout_{i}"]
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # Compile
    optimizer_dict = {"adam": Adam, "rmsprop": RMSprop, "sgd": SGD}
    model.compile(optimizer=optimizer_dict[params['optimizer']](learning_rate=params['learning_rate']),
                loss="binary_crossentropy",
                metrics=["Precision"])

    # Fit
    if (X_val is not None) and (y_val is not None):
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=5,
                                                        restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=params['batch_size'],
            callbacks=[early_stopping, ClearMemory()],
            verbose=0
        )
    else:
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss',
                                                        patience=5,
                                                        restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=params['batch_size'],
            callbacks=[early_stopping, ClearMemory()],
            verbose=0
        )
    
    return model 