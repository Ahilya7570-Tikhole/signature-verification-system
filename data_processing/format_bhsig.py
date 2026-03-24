import os
from PIL import Image
import shutil

src_root = r"d:\FL_Pro\Dataset\BHSig260-Bengali\BHSig260-Bengali"
dst_root = r"d:\FL_Pro\Dataset\BHSig260-Bengali_Formatted"

if not os.path.exists(dst_root):
    os.makedirs(dst_root)

count = 0
for i in range(1, 101):
    src_folder = os.path.join(src_root, str(i))
    if not os.path.exists(src_folder):
        continue
    
    acc_id = f"ACC{i:04d}"
    acc_folder = os.path.join(dst_root, acc_id)
    genuine_dir = os.path.join(acc_folder, 'genuine')
    forged_dir = os.path.join(acc_folder, 'forged')
    
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)
    
    for filename in os.listdir(src_folder):
        if not filename.lower().endswith('.tif'):
            continue
            
        parts = filename.split('-')
        # expected format: B-S-1-F-01.tif
        if len(parts) >= 5:
            type_str = parts[3] # 'F' or 'G'
            num_ext = parts[4] # '01.tif'
            num = num_ext.split('.')[0]
            
            dst_filename = f"{acc_id}_{type_str}_{num}.png"
            
            src_path = os.path.join(src_folder, filename)
            
            if type_str == 'G':
                dst_path = os.path.join(genuine_dir, dst_filename)
            elif type_str == 'F':
                dst_path = os.path.join(forged_dir, dst_filename)
            else:
                continue
                
            with Image.open(src_path) as img:
                img.save(dst_path, format="PNG")
                count += 1

print(f"Conversion complete. Converted {count} images.")
