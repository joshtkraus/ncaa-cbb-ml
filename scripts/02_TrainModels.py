import multiprocessing as mp

def run():
    # Libraries
    import os
    import pandas as pd
    from models.FitModels import train_models
    from models.VotingClassifier import tune_clf
    from models.PermImport import get_importance

    # Load
    data_path = os.path.join(os.path.abspath(os.getcwd()), 'data/processed/data.csv')
    data = pd.read_csv(data_path)

    # Tune Models
    split_dict = {
        2: 0.7778,
        3: 0.7778,
        4: 0.6667,
        5: 0.6667,
        6: 0.5556,
        7: 0.5556
    }
    train_models(data, split_dict)

    # Tune Voting Classifier
    tune_clf(data, split_dict)

    # Get Permutation Importance
    get_importance(data,split_dict)

if __name__ == '__main__':
    mp.freeze_support()
    run()
