# Helper worker for each round
def _tune_round(r, data, split_dict, out_q):
    from models.utils.nn import tune_nn
    from models.utils.gbm import tune_gbm

    # Tune
    nn_result = tune_nn(data, r, split_dict)
    gbm_result = tune_gbm(data, r, split_dict)

    # Return results
    out_q.put((r, nn_result, gbm_result))


def train_models(data, split_dict):
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
        p = Process(target=_tune_round, args=(r, data, split_dict, results_q))
        p.start()
        p.join()
        
        # Collect results
        round_num, nn_result, gbm_result = results_q.get()
        nn_params[round_num] = nn_result
        gbm_params[round_num] = gbm_result

    # Save
    nn_path = os.path.join(os.path.abspath(os.getcwd()), 'models/components/nn.json')
    gbm_path = os.path.join(os.path.abspath(os.getcwd()), 'models/components/gbm.json')
    with open(nn_path, 'w') as f:
        json.dump(nn_params, f)
    with open(gbm_path, 'w') as f:
        json.dump(gbm_params, f)
