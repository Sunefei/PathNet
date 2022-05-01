import torch
import torch.nn.functional as F
from tqdm import tqdm


def pre_residual_correlation(labels, model_out, label_idx):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth
    generates the initial labels used for residual correlation 
    """
    labels = labels.cpu()
    labels[labels.isnan()] = 0
    labels = labels.long()
    model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = max(labels.max().item() + 1, labels.shape[1])
    n = labels.shape[0]
    y = torch.zeros((n, c))

    if labels.shape[1] == 1:
        y[label_idx] = F.one_hot(labels[label_idx],c).float().squeeze(1) - model_out[label_idx]
    else:
        y[label_idx] = labels[label_idx].float() - model_out[label_idx]

    return y


def pre_outcome_correlation(labels, model_out, label_idx):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth
    generates the initial labels used for outcome correlation 
    """

    labels = labels.cpu()
    model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = max(labels.max().item() + 1, labels.shape[1])
    n = labels.shape[0]
    y = model_out.clone()
    if len(label_idx) > 0:
        if labels.shape[1] == 1:
            y[label_idx] = F.one_hot(labels[label_idx],c).float().squeeze(1) 
        else:
            y[label_idx] = labels[label_idx].float()
    
    return y


def general_outcome_correlation(adj, y, alpha, num_propagations, post_step, alpha_term, num_hops=1, device='cuda', display=True):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth
    general outcome correlation. alpha_term = True for outcome correlation, alpha_term = False for residual correlation 
    """
    adj = adj.to(device)
    orig_device = y.device
    y = y.to(device)
    result = y.clone()
    for _ in tqdm(range(num_propagations), disable = not display):
        for _ in range(num_hops):
            result = adj @ result
        result = alpha * result
        if alpha_term:
            result += (1-alpha)*y
        else:
            result += y
        result = post_step(result)
    return result.to(orig_device)


def double_correlation_autoscale(y_true, model_out, split_idx, A1, alpha1, num_propagations1, A2, alpha2, num_propagations2, num_hops=1, device='cuda', display=True):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth
    correct & smooth with Autoscale 
    """
    train_idx, valid_idx, test_idx = split_idx
    label_idx = torch.cat([split_idx['train']])
    residual_idx = split_idx['train']
        
    y = pre_residual_correlation(labels=y_true.data, model_out=model_out, label_idx=residual_idx)
    resid = general_outcome_correlation(adj=A1, y=y, alpha=alpha1, num_propagations=num_propagations1, 
        post_step=lambda x: torch.clamp(x, -1.0, 1.0), alpha_term=True, num_hops=num_hops, display=display, device=device)

    orig_diff = y[residual_idx].abs().sum()/residual_idx.shape[0]
    resid_scale = (orig_diff/resid.abs().sum(dim=1, keepdim=True))
    resid_scale[resid_scale.isinf()] = 1.0
    cur_idxs = (resid_scale > 1000)
    resid_scale[cur_idxs] = 1.0
    res_result = model_out + resid_scale*resid
    res_result[res_result.isnan()] = model_out[res_result.isnan()]
    y = pre_outcome_correlation(labels=y_true.data, model_out=res_result, label_idx = label_idx)
    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2, 
        post_step=lambda x: torch.clamp(x, 0,1), alpha_term=True, num_hops=num_hops, display=display, device=device)
    
    return res_result, result


def double_correlation_fixed(y_true, model_out, split_idx, A1, alpha1, num_propagations1, A2, alpha2, num_propagations2, scale=1.0, num_hops=1, device='cuda', display=True):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth
    correct & smooth with FDiff-scale 
    """
    train_idx, valid_idx, test_idx = split_idx
    label_idx = torch.cat([split_idx['train']])
    residual_idx = split_idx['train']

    y = pre_residual_correlation(labels=y_true.data, model_out=model_out, label_idx=residual_idx)
    
    fix_y = y[residual_idx].to(device)
    def fix_inputs(x):
        x[residual_idx] = fix_y
        return x
    
    resid = general_outcome_correlation(adj=A1, y=y, alpha=alpha1, num_propagations=num_propagations1, 
        post_step=lambda x: fix_inputs(x), alpha_term=True, num_hops=num_hops, display=display, device=device)
    res_result = model_out + scale*resid
    
    y = pre_outcome_correlation(labels=y_true.data, model_out=res_result, label_idx = label_idx)

    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2, 
        post_step=lambda x: x.clamp(0, 1), alpha_term=True, num_hops=num_hops, display=display, device=device)
    
    return res_result, result