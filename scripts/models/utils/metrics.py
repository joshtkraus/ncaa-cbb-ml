# Calculate Precision
def calculate_precision(predictions, actual):
    TP = sum((p == 1 and a == 1) for p, a in zip(predictions, actual))
    FP = sum((p == 1 and a == 0) for p, a in zip(predictions, actual))
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    return precision