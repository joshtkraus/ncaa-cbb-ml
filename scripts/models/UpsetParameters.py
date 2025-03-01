def tune_upset(data, split_dict, nn_feat=None, gbm_feat=None):
    print('Tuning Upset Params...')
    # Libraries
    import os
    import json
    from models.utils.upset import tune

    # Load Params
    # NN
    nn_path = os.path.join(os.path.abspath(os.getcwd()), 'models/pre_fs/nn.json')
    with open(nn_path, "r") as json_file:
        nn_params = json.load(json_file)
    nn_params = {int(key): value for key, value in nn_params.items()}
    # GBM
    gbm_path = os.path.join(os.path.abspath(os.getcwd()), 'models/pre_fs/gbm.json')
    with open(gbm_path, "r") as json_file:
        gbm_params = json.load(json_file)
    gbm_params = {int(key): value for key, value in gbm_params.items()}

    # Load Weights
    weights_path = os.path.join(os.path.abspath(os.getcwd()), 'models/weights.json')
    with open(weights_path, "r") as json_file:
        weights = json.load(json_file)
    weights = {int(key): value for key, value in weights.items()}

    # Get Upset Parameters
    upset_params = tune(data, split_dict, nn_params, gbm_params, weights, nn_feat, gbm_feat)
    
    # Export Upset Parameters
    upset_path = os.path.join(os.path.abspath(os.getcwd()), 'models/upset_params.json')
    with open(upset_path, 'w') as f:
        json.dump(upset_params, f)