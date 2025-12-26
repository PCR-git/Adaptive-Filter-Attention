import numpy as np
import torch
    
##########################################################################################
##########################################################################################

def apply_weight_masks(module, lambda_h_v, args):

    weight_mask_in = torch.zeros(args.d_e*2, args.d_e).to(args.device)
    weight_mask_in[0:2,0:2] = 1
    weight_mask_in[128:130, :] = 1

    weight_mask_out = torch.zeros(args.d_e, args.d_e*2).to(args.device)
    weight_mask_out[0:2,0:2] = 1
    weight_mask_out[:, 128:130] = 1

    bias_mask_in = torch.zeros(args.d_e*2).to(args.device)
    bias_mask_out = torch.zeros(args.d_e).to(args.device)

    with torch.no_grad():
        module.W_q.weight *= weight_mask_in
        module.W_k.weight *= weight_mask_in
        module.W_v.weight *= weight_mask_in
        module.W_o.weight *= weight_mask_out

        module.W_q.bias *= bias_mask_in
        module.W_k.bias *= bias_mask_in
        module.W_v.bias *= bias_mask_in
        module.W_o.bias *= bias_mask_out
        
        lambda_h_v[0] = lambda_h_v[0]*0 - 0.1
        lambda_h_v[1,:,0] = -1.0
        lambda_h_v[1,:,1] = 1.0
        lambda_h_v[1] = lambda_h_v[1]/torch.abs(lambda_h_v[1])
        
#         lambda_sigma_v = 0.0
#         lambda_eta_v = 1.0
#         lambda_gamma_v = 0.0

    print('Weight masking on.')

    return lambda_h_v
