def set_seed(seed=23):
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def create_model_nn(input_shape, params):
    from tensorflow import keras
    from tensorflow.keras import layers

    set_seed()

    # Create Model
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    # Number of layers
    num_layers = params['num_layers']
    for i in range(num_layers):
        num_units = params[f"units_{i}"]
        activation = params[f"activation_{i}"]

        model.add(layers.Dense(num_units, activation=activation))
        
        # BatchNormalization
        if params[f"batch_norm_{i}"]:
            model.add(layers.BatchNormalization())
        
        # Dropout Rate
        dropout_rate = params[f"dropout_{i}"]
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation="sigmoid"))

    return model    

import tensorflow as tf
class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        import gc
        gc.collect()
def objective_nn(trial, X_train, X_val, y_train, y_val, params):
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD
    from tensorflow.keras import backend as K
    from sklearn.metrics import average_precision_score
    import gc

    # Mask for Feature Selection
    num_features = X_train.shape[1]
    feature_mask = np.array([trial.suggest_categorical(f"feature_{i}", [0, 1]) for i in range(num_features)])
    # Check that at least 1 feature selected
    if feature_mask.sum() == 0:
        # Penalty for removing all features
        return float("inf")
    # Feature Selection
    X_train_sel = X_train[:, feature_mask == 1]
    X_val_sel = X_val[:, feature_mask == 1]

    # Create Model
    model = create_model_nn(X_train_sel.shape[1], params)
    optimizer_dict = {"adam": Adam, "rmsprop": RMSprop, "sgd": SGD}
    model.compile(optimizer=optimizer_dict[params['optimizer']](learning_rate=params['learning_rate']),
                loss="binary_crossentropy",
                metrics=["Precision"])
    
     # Early Stopping & Pruning
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    restore_best_weights=True)

    # Fit
    model.fit(
        X_train_sel, y_train,
        validation_data=(X_val_sel, y_val),
        epochs=50,
        batch_size=params['batch_size'],
        callbacks=[early_stopping, ClearMemory()],
        verbose=0
    )
    pred_prob = model.predict(X_val_sel, verbose=0)
    pred = pred_prob > 0.5
    pred = pred.astype(int).flatten()

    # Clear Memory
    K.clear_session()
    gc.collect()
    del model

    return -average_precision_score(y_val, pred)

def objective_gbm(trial, X_train, X_val, y_train, y_val, params):
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import average_precision_score
    set_seed()

    # Mask for Feature Selection
    num_features = X_train.shape[1]
    feature_mask = np.array([trial.suggest_categorical(f"feature_{i}", [0, 1]) for i in range(num_features)])
    # Check that at least 1 feature selected
    if feature_mask.sum() == 0:
        # Penalty for removing all features
        return float("inf")
    # Feature Selection
    X_train_sel = X_train[:, feature_mask == 1]
    X_val_sel = X_val[:, feature_mask == 1]
    
    # Subset Params
    params_sub = {key: value for key, value in params.items() if key not in ['num_boost_round']}
    # Train the model
    dtrain = xgb.DMatrix(X_train_sel, label=y_train)
    dval = xgb.DMatrix(X_val_sel, label=y_val)
    model = xgb.train(
        params_sub, 
        dtrain, 
        num_boost_round=params['num_boost_round'],
        evals=[(dval, "validation")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    pred = model.predict(dval)
    return -average_precision_score(y_val, pred)

def feat_sel_nn(data, r, split_dict, params, n_trials=300):
    import os
    import numpy as np
    from models.utils.DataProcessing import create_splits, get_modeling_cols
    import optuna
    from optuna.visualization import plot_optimization_history
    from itertools import compress
    import tensorflow as tf

    # Supress Logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Set Memory Allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Columns
    features = get_modeling_cols(data)
    
    # Create Data
    X_SMTL, y_SMTL = create_splits(data, r, train=True)
    X, y = create_splits(data, r, train=False)

    # Data Splits
    split_idx = int(split_dict[r] * len(X))
    split_idx_SMTL = np.where((X_SMTL == X[split_idx]).all(axis=1))[0][0]
    X_train, X_val = X_SMTL[:split_idx_SMTL], X[split_idx:]
    y_train, y_val = y_SMTL[:split_idx_SMTL], y[split_idx:]

    # Tuning
    study = optuna.create_study(
        study_name=f"feature_selection_nn_round_{r}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=23)
    )
    study.optimize(lambda trial: objective_nn(trial, X_train, X_val, y_train, y_val, params), 
                   n_trials=n_trials,
                   gc_after_trial=True)
    
    # Save Plot
    fig = plot_optimization_history(study)
    path = os.path.join(os.path.abspath(os.getcwd()), f"results/models/feat_sel/nn/round_{r}.png")
    fig.write_image(path)
    
    # Get Selected Columns
    best_mask = [study.best_params[f"feature_{i}"] for i in range(X.shape[1])]
    selected_features = list(compress(features, best_mask))
    return selected_features

def feat_sel_gbm(data, r, split_dict, params, n_trials=600):
    import os
    import numpy as np
    from models.utils.DataProcessing import create_splits, get_modeling_cols
    import optuna
    from optuna.visualization import plot_optimization_history
    from itertools import compress

    # Supress Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Columns
    features = get_modeling_cols(data)
    
    # Create Data
    X_SMTL, y_SMTL = create_splits(data, r, train=True)
    X, y = create_splits(data, r, train=False)

    # Data Splits
    split_idx = int(split_dict[r] * len(X))
    split_idx_SMTL = np.where((X_SMTL == X[split_idx]).all(axis=1))[0][0]
    X_train, X_val = X_SMTL[:split_idx_SMTL], X[split_idx:]
    y_train, y_val = y_SMTL[:split_idx_SMTL], y[split_idx:]

    # Tuning
    study = optuna.create_study(
        study_name=f"feature_selection_gbm_round_{r}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=23)
    )
    study.optimize(lambda trial: objective_gbm(trial, X_train, X_val, y_train, y_val, params), 
                   n_trials=n_trials,
                   gc_after_trial=True)
    
    # Save Plot
    fig = plot_optimization_history(study)
    path = os.path.join(os.path.abspath(os.getcwd()), f"results/models/feat_sel/gbm/round_{r}.png")
    fig.write_image(path)
    
    # Get Selected Columns
    best_mask = [study.best_params[f"feature_{i}"] for i in range(X.shape[1])]
    selected_features = list(compress(features, best_mask))
    return selected_features