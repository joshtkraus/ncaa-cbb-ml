# Tune Models
def get_importance(data, split_dict):
    print('Calculating Permutation Importance...')
    # Libraries
    import os
    import json
    from models.utils.importance import get_importance

    # Load Params
    # NN
    nn_path = os.path.join(os.path.abspath(os.getcwd()), 'models/components/nn.json')
    with open(nn_path, "r") as json_file:
        nn_params = json.load(json_file)
    nn_params = {int(key): value for key, value in nn_params.items()}
    # GBM
    gbm_path = os.path.join(os.path.abspath(os.getcwd()), 'models/components/gbm.json')
    with open(gbm_path, "r") as json_file:
        gbm_params = json.load(json_file)
    gbm_params = {int(key): value for key, value in gbm_params.items()}
    # Weights
    weights_path = os.path.join(os.path.abspath(os.getcwd()), 'models/weights.json')
    with open(weights_path, "r") as json_file:
        weights = json.load(json_file)
    weights = {int(key): value for key, value in weights.items()}

    # Get Weights
    get_importance(data, split_dict, nn_params, gbm_params, weights)