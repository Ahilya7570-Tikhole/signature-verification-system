import os
import shutil

src_root = r"d:\FL_Pro\Datasets\CEDAR\CEDAR"
dst_root = r"d:\FL_Pro\Dataset\CEDAR"

if not os.path.exists(dst_root):
    os.makedirs(dst_root)

count = 0
for i in range(1, 56):
    src_folder = os.path.join(src_root, str(i))
    if not os.path.exists(src_folder):
        continue
    
    # Calculate ACC ID starting from 261 => writer 1 is ACC0261
    acc_id_num = 260 + i
    acc_id = f"ACC{acc_id_num:04d}"
    
    acc_folder = os.path.join(dst_root, acc_id)
    genuine_dir = os.path.join(acc_folder, 'genuine')
    forged_dir = os.path.join(acc_folder, 'forged')
    
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)
    
    for filename in os.listdir(src_folder):
        if not filename.lower().endswith('.png'):
            continue
            
        # Example format: original_1_1.png or forgeries_1_1.png
        # We need to parse this properly
        parts = filename.split('_')
        if len(parts) >= 3:
            type_str_full = parts[0] # 'original' or 'forgeries'
            num_ext = parts[2] # '1.png'
            num = num_ext.split('.')[0]
            num_padded = f"{int(num):02d}"
            
            src_path = os.path.join(src_folder, filename)
            
            if type_str_full.lower() == 'original':
                dst_filename = f"{acc_id}_G_{num_padded}.png"
                dst_path = os.path.join(genuine_dir, dst_filename)
                shutil.copy2(src_path, dst_path)
                count += 1
            elif type_str_full.lower() == 'forgeries':
                dst_filename = f"{acc_id}_F_{num_padded}.png"
                dst_path = os.path.join(forged_dir, dst_filename)
                shutil.copy2(src_path, dst_path)
                count += 1

print(f"Formatting complete. Processed {count} images in {dst_root}.")
