from sklearn.metrics import confusion_matrix
def metrics(true_series, pred_series):

    """
        from Spyders.test_function import metrics
        TN, FN, FP, TP, a, b = metrics(biến đầu vào 1, biến đầu vào 2, THRESHOLD_GLOBAL)
        ***Chú ý biến đầu vào 1 và 2 đều dạng Series
    """
    confusion_mat = confusion_matrix(true_series.astype(int), pred_series.astype(int)).ravel()

    precision = confusion_mat[3] / (confusion_mat[2]+confusion_mat[3]) # class 1 false ratio

    FOR = confusion_mat[1] / (confusion_mat[1]+confusion_mat[3]) # class 0 false ratio
    return confusion_mat[0], confusion_mat[1], confusion_mat[2], confusion_mat[3], precision, FOR