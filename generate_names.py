from tensorflow.keras.models import load_model

model = load_model("model/plant_model.h5", compile=False)

num = model.output_shape[1]
print("Model classes =", num)

# generate generic names
names = [f"Class_{i+1}" for i in range(num)]

with open("model/class_names.txt", "w", encoding="utf-8") as f:
    for n in names:
        f.write(n + "\n")

print("Saved class_names.txt with", num, "entries.")