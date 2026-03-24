import os
import shutil

# Mappings
banks = {
    "Bank1": {
        "Bengali": (1, 35),
        "Hindi": (101, 155),
        "CEDAR": (261, 275)
    },
    "Bank2": {
        "Bengali": (36, 70),
        "Hindi": (156, 210),
        "CEDAR": (276, 290)
    },
    "Bank3": {
        "Bengali": (71, 100),
        "Hindi": (211, 260),
        "CEDAR": (291, 315)
    }
}

dataset_paths = {
    "Bengali": r"d:\FL_Pro\Dataset\BHSig260-Bengali",
    "Hindi": r"d:\FL_Pro\Dataset\BHSig260-Hindi",
    "CEDAR": r"d:\FL_Pro\Dataset\CEDAR"
}

root_dir = r"d:\FL_Pro"

count_total = 0

for bank, sets in banks.items():
    local_dir = os.path.join(root_dir, bank, "Local_dataset")
    os.makedirs(local_dir, exist_ok=True)
    
    count_bank = 0
    for ds_name, (start, end) in sets.items():
        src_ds_dir = dataset_paths[ds_name]
        for acc_num in range(start, end + 1):
            acc_id = f"ACC{acc_num:04d}"
            src_acc_dir = os.path.join(src_ds_dir, acc_id)
            dst_acc_dir = os.path.join(local_dir, acc_id)
            
            if os.path.exists(src_acc_dir):
                shutil.copytree(src_acc_dir, dst_acc_dir, dirs_exist_ok=True)
                count_bank += 1
                count_total += 1
            else:
                print(f"Warning: {acc_id} not found in {src_ds_dir}")
                
    print(f"Finished copying {count_bank} accounts to {bank}/Local_dataset")

print(f"Total accounts copied: {count_total}")
