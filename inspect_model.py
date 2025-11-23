import os

train_dir = "dataset/train"

if os.path.exists(train_dir):
    # Get all folder names (these are your classes)
    folders = sorted([f for f in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, f))])
    
    print(f"Found {len(folders)} classes in dataset/train folder")
    print("="*60)
    
    # Save to file
    with open("real_class_names.txt", "w", encoding="utf-8") as f:
        for i, folder in enumerate(folders):
            f.write(folder + "\n")
            print(f"{i}: {folder}")
    
    print("="*60)
    print(f"✅ Saved all {len(folders)} class names to real_class_names.txt")
    
else:
    print(f"❌ Folder not found: {train_dir}")
    print("\nPlease check if your dataset folder exists and contains the train subfolder.")