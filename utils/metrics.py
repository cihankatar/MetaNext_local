from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import torch
import numpy as np

def IoU(y_true, y_pred):
    
    intersection = (y_true * y_pred).sum()
    union = (y_true).sum() + (y_pred).sum() - intersection
    
    return torch.mean( (intersection + torch.tensor(1e-6)) / (union +  torch.tensor(1e-6)))

def calculate_metrics(y_true, y_pred):

    y_true = y_true.numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard   = jaccard_score(y_true, y_pred)
    score_f1        = f1_score(y_true, y_pred)
    score_recall    = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc       = accuracy_score(y_true, y_pred)
    score_IoU       = IoU(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_IoU]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask
