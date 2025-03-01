# Helper worker for each round
def _tune_round(r, data, split_dict, nn_params, gbm_params, out_q):
    from models.utils.feat_sel import feat_sel_nn, feat_sel_gbm

    # Feature Selection
    nn_feat_r = feat_sel_nn(data, r, split_dict, nn_params[r])
    gbm_feat_r = feat_sel_gbm(data, r, split_dict, gbm_params[r])

    # Return results
    out_q.put((r, nn_feat_r, gbm_feat_r))

def select_features(data, split_dict):
    import os
    import json
    from multiprocessing import Process, Queue

    # Get Tuned Params
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

    # Initialize
    split_dict = {
        2: 0.8,
        3: 0.8,
        4: 0.7,
        5: 0.7,
        6: 0.6,
        7: 0.6
    }
    nn_feat = {}
    gbm_feat = {}
    results_q = Queue()

    print('Feature Selection...')
    for r in range(2,8):
        print('Round '+str(r))

        # Subprocess
        p = Process(target=_tune_round, args=(r, data, split_dict, nn_params, gbm_params, results_q))
        p.start()
        p.join()
        
        # Collect results
        round_num, nn_feat_r, gbm_feat_r = results_q.get()
        nn_feat[round_num] = nn_feat_r
        gbm_feat[round_num] = gbm_feat_r
    
    # Export Features
    # NN
    nn_path = os.path.join(os.path.abspath(os.getcwd()), 'models/features/nn.json')
    with open(nn_path, 'w') as f:
       json.dump(nn_feat, f)
    # GBM
    gbm_path = os.path.join(os.path.abspath(os.getcwd()), 'models/features/gbm.json')
    with open(gbm_path, 'w') as f:
        json.dump(gbm_feat, f)