"""Iterative permutation-based feature selection for NN and GBM models."""


def select_features(data, split_dict):
    """Iteratively drop features with zero or negative permutation importance.

    For each round, fits NN and GBM on the current feature subset, computes
    permutation importance for each model independently, and removes any
    feature whose mean loss increase is <= 0. Repeats until no features are
    removed in a full pass. NN and GBM maintain separate surviving feature
    sets throughout. The final sets are saved to models/components/features.json.

    Args:
        data: Full modeling DataFrame.
        split_dict: Dict mapping round number to train/val split ratio.

    Returns:
        features_dict: Nested dict keyed by round number (int), each value a
            dict with keys 'nn' and 'gbm' containing lists of surviving
            feature name strings.
    """
    import json
    import os

    import numpy as np

    from models.utils.DataProcessing import create_splits
    from models.utils.importance import _round_importances

    print("Selecting Features...")

    nn_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/nn.json")
    with open(nn_path, "r") as f:
        nn_params = json.load(f)
    nn_params = {int(k): v for k, v in nn_params.items()}

    gbm_path = os.path.join(os.path.abspath(os.getcwd()), "models/components/gbm.json")
    with open(gbm_path, "r") as f:
        gbm_params = json.load(f)
    gbm_params = {int(k): v for k, v in gbm_params.items()}

    rng = np.random.RandomState(23)
    features_dict = {}

    for r in range(2, 8):
        print(f"  Round {r}")

        # Initialise surviving sets from the full feature list
        all_features = create_splits(data, r, get_features=True)
        surviving_nn = list(all_features)
        surviving_gbm = list(all_features)

        iteration = 0
        while True:
            iteration += 1
            drop_nn = [f for f in all_features if f not in set(surviving_nn)] or None
            drop_gbm = [f for f in all_features if f not in set(surviving_gbm)] or None

            nn_imp, gbm_imp, nn_features, gbm_features, _, _ = _round_importances(
                data,
                r,
                split_dict,
                nn_params[r],
                gbm_params[r],
                rng,
                drop_cols_nn=drop_nn,
                drop_cols_gbm=drop_gbm,
            )

            # Features to drop this iteration: those with importance <= 0
            nn_drop = {f for f, imp in zip(nn_features, nn_imp, strict=False) if imp <= 0}
            gbm_drop = {f for f, imp in zip(gbm_features, gbm_imp, strict=False) if imp <= 0}

            new_surviving_nn = [f for f in surviving_nn if f not in nn_drop]
            new_surviving_gbm = [f for f in surviving_gbm if f not in gbm_drop]

            n_dropped = (
                len(surviving_nn)
                - len(new_surviving_nn)
                + len(surviving_gbm)
                - len(new_surviving_gbm)
            )
            print(
                f"    Iteration {iteration}: dropped {len(surviving_nn) - len(new_surviving_nn)}"
                f" NN features, {len(surviving_gbm) - len(new_surviving_gbm)} GBM features"
            )

            surviving_nn = new_surviving_nn
            surviving_gbm = new_surviving_gbm

            if n_dropped == 0:
                break

        features_dict[r] = {"nn": surviving_nn, "gbm": surviving_gbm}
        print(
            f"  Round {r} final: {len(surviving_nn)} NN features, "
            f"{len(surviving_gbm)} GBM features "
            f"(started with {len(all_features)})"
        )

    path = os.path.join(os.path.abspath(os.getcwd()), "models/components/features.json")
    # JSON keys must be strings; convert int round keys before saving
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in features_dict.items()}, f, indent=2)

    return features_dict
