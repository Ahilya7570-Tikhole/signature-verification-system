import os
import torch
import shutil
from federated_averaging import federated_avg

def main():
    # Define project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Paths to local models
    local_model_paths = [
        os.path.join(project_root, "Bank1", "local_model.pth"),
        os.path.join(project_root, "Bank2", "local_model.pth"),
        os.path.join(project_root, "Bank3", "local_model.pth")
    ]
    
    # Verify all models exist
    available_models = [path for path in local_model_paths if os.path.exists(path)]
    
    if len(available_models) < 1:
        print("Error: No local models found for averaging.")
        return
    
    print(f"Averaging {len(available_models)} models...")
    
    # Run Federated Averaging
    global_state_dict = federated_avg(available_models)
    
    if global_state_dict is None:
        print("Error: Federated averaging failed.")
        return
    
    # Save the global model
    server_dir = os.path.dirname(os.path.abspath(__file__))
    saved_models_dir = os.path.join(server_dir, "saved_models")
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)
        
    global_model_path = os.path.join(saved_models_dir, "global_model.pth")
    torch.save(global_state_dict, global_model_path)
    print(f"Global model saved to {global_model_path}")
    
    # Send global model back to local banks
    banks = ["Bank1", "Bank2", "Bank3"]
    for bank in banks:
        bank_path = os.path.join(project_root, bank, "global_model.pth")
        shutil.copy2(global_model_path, bank_path)
        print(f"Global model sent back to {bank}: {bank_path}")

if __name__ == "__main__":
    main()
