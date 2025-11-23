import os

train = "dataset/train"

folders = sorted([f for f in os.listdir(train) if os.path.isdir(os.path.join(train, f))])

with open("real_class_names.txt", "w", encoding="utf-8") as f:
    for name in folders:
        f.write(name + "\n")

print("Saved real_class_names.txt with", len(folders), "classes.")
