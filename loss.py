from utils import *

def SSIMLoss(pred, target) :
    # Compute Loss
    loss = calcSSIM(target, pred)
    
    return -loss