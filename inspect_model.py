from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("model/plant_model.h5", compile=False)
print("Model loaded.")

# Method 1: Check full model output shape
try:
    out_shape = model.output_shape
    print("Model output_shape:", out_shape)
    if len(out_shape) == 2:
        print("Number of classes:", out_shape[1])
    else:
        print("Model output not flat classification:", out_shape)
except Exception as e:
    print("Failed reading model.output_shape:", e)

# Method 2: Directly check output layer node count
try:
    last = model.layers[-1]
    print("\nLast layer:", last.name)
    config = last.get_config()
    if "units" in config:
        print("Last Dense units:", config["units"])
    else:
        print("Layer config has no 'units' key. Full config:")
        print(config)
except Exception as e:
    print("Could not inspect last layer:", e)
