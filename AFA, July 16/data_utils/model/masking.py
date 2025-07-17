import numpy as np
import torch

##########################################################################################
##########################################################################################

def init_weight_masks(module, args):
    """
    Create masks for parameter matrices (used for testing)
    """
    module.eigen_mask = torch.zeros(2,int(args.d_v/2),1).to(args.device)
    module.eigen_mask[:,0:1,:] = 1

    module.noise_mask = torch.zeros(args.d_v,1).to(args.device)
    module.noise_mask[0:2,:] = 1

#     module.weight_mask_k = torch.zeros_like(module.W_k).to(args.device)
    module.weight_mask_k = torch.zeros(args.d_k, args.d_e).to(args.device)
    module.weight_mask_k[0:2,0:2] = 1

#     module.weight_mask_v = torch.zeros_like(module.W_v).to(args.device)
    module.weight_mask_v = torch.zeros(args.d_v, args.d_e).to(args.device)
    module.weight_mask_v[0:2,0:2] = 1

#     module.weight_mask_p = torch.zeros_like(module.W_p).to(args.device)
    module.weight_mask_p = torch.zeros(args.d_v, args.d_e).to(args.device)
    module.weight_mask_p[0:2,0:2] = 1
    
##########################################################################################
##########################################################################################

def apply_weight_masks(module, args):
    """
    Mask out all eigenvalues & weights above a certain dimension
    """
    
    if args.weight_mask == 1:
        # Mask out eigenvalues:
        lambda_h = module.lambda_h * module.eigen_mask  
        # Mask out noise covariance eigenvals
        lambda_Omega = module.lambda_Omega * module.noise_mask
        lambda_Gamma = module.lambda_Gamma * module.noise_mask

        # Mask weight values
        W_q = module.W_q * module.weight_mask_k
        W_k = module.W_k * module.weight_mask_k
        W_v = module.W_v * module.weight_mask_v
        W_p = module.W_p * module.weight_mask_p
        W_r = module.W_r * module.weight_mask_v
        W_e = module.W_e * module.weight_mask_p

        # Mask biases
        W_q_b = module.W_q_b * module.weight_mask_k[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_k_b = module.W_k_b * module.weight_mask_k[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_v_b = module.W_v_b * module.weight_mask_v[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_p_b = module.W_p_b * module.weight_mask_p[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_r_b = module.W_r_b * module.weight_mask_v[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_e_b = module.W_e_b * module.weight_mask_p[:,:,:,0].unsqueeze(0).unsqueeze(-1)
    else:
        lambda_h = module.lambda_h
        lambda_Omega = module.lambda_Omega
        lambda_Gamma = module.lambda_Gamma

        W_q = module.W_q
        W_k = module.W_k
        W_v = module.W_v
        W_p = module.W_p
        W_r = module.W_r
        W_e = module.W_e

        W_q_b = module.W_q_b
        W_k_b = module.W_k_b
        W_v_b = module.W_v_b
        W_p_b = module.W_p_b
        W_r_b = module.W_r_b
        W_e_b = module.W_e_b
        
    return lambda_h, lambda_Omega, lambda_Gamma, W_q, W_k, W_v, W_p, W_r, W_e, W_q_b, W_k_b, W_v_b, W_p_b, W_r_b, W_e_b

##########################################################################################
##########################################################################################

def apply_net_weight_masks(module_instance):
    """
    Applies weight and bias masks to the relevant complex linear layers and
    dynamic parameters (lambda, Omega, Gamma) *within a single given module instance*.
    It does not recurse into sub-modules.

    Args:
        module_instance (torch.nn.Module): The specific module instance (e.g., a FullPrecisionAttentionBlockShared layer,
                                           or the main PrecisionAttentionBlock_Nlayer module) to apply masks to.
    """
    if not hasattr(module_instance, 'args') or module_instance.args.weight_mask == 0:
        # If the module doesn't have args or weight_mask is not enabled, do nothing.
        return

    with torch.no_grad():
        # Define lists of (attribute_name, mask_attribute_name) for linear layers
        # The mask_attribute_name should correspond to the type of mask needed (k, v, or p)
        linear_layer_configs = [
            ('W_q', 'weight_mask_k'),
            ('W_k', 'weight_mask_k'),
            ('shared_W_q', 'weight_mask_k'),
            ('shared_W_k', 'weight_mask_k'),
            ('W_v', 'weight_mask_v'),
            ('W_p', 'weight_mask_p'),
            # Add W_r, W_e here if you uncomment them in your model
            # ('W_r', 'weight_mask_v'),
            # ('W_e', 'weight_mask_p'),
        ]

        # Apply masks and zero biases for linear layers
        for layer_attr, mask_attr in linear_layer_configs:
            if hasattr(module_instance, layer_attr) and hasattr(module_instance, mask_attr):
                layer = getattr(module_instance, layer_attr)
                mask = getattr(module_instance, mask_attr)
                apply_complex_mask(layer, mask)
                zero_complex_bias(layer)

        # Define lists of (parameter_name, mask_attribute_name) for dynamic parameters
        dynamic_param_configs = [
            ('lambda1', 'eigen_mask'),
            ('shared_lambda1', 'eigen_mask'), # Added for consistency with naming
            ('lambda_Omega', 'noise_mask'),
            ('lambda_Gamma', 'noise_mask'),
        ]

        # Apply masks for dynamic parameters
        for param_attr, mask_attr in dynamic_param_configs:
            if hasattr(module_instance, param_attr) and hasattr(module_instance, mask_attr):
                param = getattr(module_instance, param_attr)
                mask = getattr(module_instance, mask_attr)
                param.data *= mask.data # Use .data to directly modify the parameter tensor

        # Handle separate parameters for keys if enabled
        if hasattr(module_instance, 'args') and module_instance.args.sep_params == 1:
            separate_key_param_configs = [
                ('lambda1_k', 'eigen_mask_k'),
                ('lambda_Omega_k', 'noise_mask_k'),
                ('lambda_Gamma_k', 'noise_mask_k'),
            ]
            for param_attr, mask_attr in separate_key_param_configs:
                if hasattr(module_instance, param_attr) and hasattr(module_instance, mask_attr):
                    param = getattr(module_instance, param_attr)
                    mask = getattr(module_instance, mask_attr)
                    param.data *= mask.data # Use .data to directly modify the parameter tensor
                    
##########################################################################################
##########################################################################################
    
def apply_complex_mask(layer, mask):
    """
    Permanently apply a binary mask to the real and imaginary weights of a ComplexLinear layer.
    """
    with torch.no_grad():
        layer.real.weight *= mask.squeeze()
        layer.imag.weight *= mask.squeeze()

def zero_complex_bias(layer):
    """
    Permanently zeroes out the biases of a ComplexLinear layer.
    """
    with torch.no_grad():
        layer.real.bias.zero_()
        layer.imag.bias.zero_()


