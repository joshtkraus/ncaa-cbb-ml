def custom_time_series_split(n_samples, n_splits, train_size, test_size, gap):
    """
    Generator for custom time-series splits.

    Parameters:
    - n_samples: Total number of data points.
    - n_splits: Number of splits.
    - train_size: Number of training samples in each fold.
    - test_size: Number of testing samples in each fold.
    - gap: Number of data points skipped between each fold.

    Yields:
    - train_index: Indices for the training data (as a list).
    - test_index: Indices for the test data (as a list).
    """
    start = 0
    for _ in range(n_splits):
        train_start = start
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > n_samples:
            break

        train_index = list(range(train_start, train_end))
        test_index = list(range(test_start, test_end))

        yield train_index, test_index

        start += gap