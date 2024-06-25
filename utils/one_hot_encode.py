
import torch

def one_hot(targets, n_classes):
    batch_size,h,w=targets.shape
    targets=torch.flatten(input=targets)
    label_zero = 0

    target_one_hot   = torch.zeros(batch_size*h*w,n_classes)

    for i in range(torch.tensor(targets.shape)):
            
            if targets[i]==label_zero:
                target_one_hot[i] = torch.tensor([1,0])
            else:
                target_one_hot[i] = torch.tensor([0,1])
                
    target_one_hot=target_one_hot.reshape(batch_size,h,w,n_classes)
    
    return target_one_hot 


def label_encode(targets):
    batch_size,h,w = targets.shape
    label_zero = 0

    for idx, label in enumerate(targets):
        for i in range(h):
            for j in range (w):
                if targets[idx,j,i]>=label_zero and targets[idx,j,i]<=0.5:
                    targets[idx,j,i] = torch.tensor([0])
                else:
                    targets[idx,j,i] = torch.tensor([1])
    return targets