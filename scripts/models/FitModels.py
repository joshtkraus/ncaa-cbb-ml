# Helper worker for each round
def _tune_round(r, data, split_dict, nn_feat, gbm_feat, out_q):
    from models.utils.nn import tune_nn
    from models.utils.gbm import tune_gbm

    # Tune
    if nn_feat is None or gbm_feat is None:
        nn_result = tune_nn(data, r, split_dict)
        gbm_result = tune_gbm(data, r, split_dict)
    else:
        nn_result = tune_nn(data, r, split_dict, nn_feat[r])
        gbm_result = tune_gbm(data, r, split_dict, gbm_feat[r])

    # Return results
    out_q.put((r, nn_result, gbm_result))


def train_models(data, split_dict, nn_feat=None, gbm_feat=None):
    import os
    import json
    from multiprocessing import Process, Queue
    print('Tuning Models...')
    
    # Initialize
    nn_params = {}
    gbm_params = {}
    results_q = Queue()

    for r in range(2, 8):
        print('Round', r)

        # Subprocess
        p = Process(target=_tune_round, args=(r, data, split_dict, nn_feat, gbm_feat, results_q))
        p.start()
        p.join()
        
        # Collect results
        round_num, nn_result, gbm_result = results_q.get()
        nn_params[round_num] = nn_result
        gbm_params[round_num] = gbm_result

    # Save
    if nn_feat is None or gbm_feat is None:
        nn_path = os.path.join(os.path.abspath(os.getcwd()), 'models/pre_fs/nn.json')
        gbm_path = os.path.join(os.path.abspath(os.getcwd()), 'models/pre_fs/gbm.json')
    else:
        nn_path = os.path.join(os.path.abspath(os.getcwd()), 'models/post_fs/nn.json')
        gbm_path = os.path.join(os.path.abspath(os.getcwd()), 'models/post_fs/gbm.json')

    with open(nn_path, 'w') as f:
        json.dump(nn_params, f)
    with open(gbm_path, 'w') as f:
        json.dump(gbm_params, f)
