import torch
import copy

def federated_avg(local_model_paths):
    """
    Averages the weights of multiple local models.
    
    Args:
        local_model_paths (list): List of paths to local model files (.pth)
    
    Returns:
        dict: The averaged state dictionary.
    """
    if not local_model_paths:
        return None
    
    # Load the first model to get the structure
    global_state_dict = torch.load(local_model_paths[0], map_location=torch.device('cpu'))
    
    # Initialize a copy for averaging
    averaged_state_dict = copy.deepcopy(global_state_dict)
    
    # Sum weights from other models
    for path in local_model_paths[1:]:
        local_state_dict = torch.load(path, map_location=torch.device('cpu'))
        for key in averaged_state_dict:
            averaged_state_dict[key] += local_state_dict[key]
            
    # Divide by number of models
    num_models = len(local_model_paths)
    for key in averaged_state_dict:
        averaged_state_dict[key] = averaged_state_dict[key] / num_models
        
    return averaged_state_dict
