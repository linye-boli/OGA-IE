import torch 
import os 

def relative_l2(upred, uref):
    return torch.linalg.norm(upred - uref) / torch.linalg.norm(uref)

def init_records(log_root, task_nm, exp_nm):
    exp_root = os.path.join(log_root, task_nm, exp_nm)
    os.makedirs(exp_root, exist_ok=True)

    log_outpath = os.path.join(exp_root, 'log.npy')
    upred_outpath = os.path.join(exp_root, 'upred.npy')
    model_outpath = os.path.join(exp_root, 'model.npy')
    
    return log_outpath, upred_outpath, model_outpath