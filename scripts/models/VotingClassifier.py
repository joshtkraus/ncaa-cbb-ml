# Tune Models
def tune_clf(data, split_dict):
    print('Tuning Weights...')
    # Libraries
    import os
    import json
    from models.utils.voting_clf import tune_weights

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

    # Get Weights
    weights = tune_weights(data, split_dict, nn_params, gbm_params)

    # Export Weights
    weights_path = os.path.join(os.path.abspath(os.getcwd()), 'models/weights.json')
    with open(weights_path, 'w') as f:
        json.dump(weights, f)