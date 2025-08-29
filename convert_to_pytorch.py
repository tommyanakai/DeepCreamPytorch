import tensorflow as tf
import torch
import numpy as np
# import model
# -------------------------------
# 1. Load TensorFlow checkpoint
# -------------------------------
ckpt_path = r"C:\Tools\DeepCreamPy\models\bar\Train_775000"  # adjust path
# ckpt_path=r"C:\Tools\DeepCreamPy\models\mosaic\Train_290000"
reader = tf.train.NewCheckpointReader(ckpt_path)
var_to_shape_map = reader.get_variable_to_shape_map()

mapping = {
    "CB1/ML/biases": "contextual_block.fuse_conv.bias",
    "CB1/ML/weights": "contextual_block.fuse_conv.weight",
    "G_de/Conv_3/biases": "decoder.dl2.conv2.bias",
    "G_de/Conv/biases": "decoder.dl1.conv1.bias",
    "G_de/Conv/weights": "decoder.dl1.conv1.weight",
    "G_de/Conv_1/biases": "decoder.dl1.conv2.bias",
    "G_de/Conv_5/weights": "decoder.dl3.conv2.weight",
    "G_de/Conv_5/biases": "decoder.dl3.conv2.bias",
    "G_de/Conv_1/weights": "decoder.dl1.conv2.weight",
    "G_de/Conv_2/biases": "decoder.dl2.conv1.bias",
    "G_de/Conv_4/biases": "decoder.dl3.conv1.bias",
    "G_de/Conv_2/weights": "decoder.dl2.conv1.weight",
    "G_de/Conv_8/weights": "decoder.out_conv.weight",
    "G_de/Conv_7/weights": "decoder.dl4.conv2.weight",
    "G_de/Conv_3/weights": "decoder.dl2.conv2.weight",
    "G_de/Conv_4/weights": "decoder.dl3.conv1.weight",
    "G_de/Conv_8/biases": "decoder.out_conv.bias",
    "G_de/Conv_6/biases": "decoder.dl4.conv1.bias",
    "G_de/Conv_6/weights": "decoder.dl4.conv1.weight",
    "G_de/Conv_7/biases": "decoder.dl4.conv2.bias",
    "G_en/Conv/biases": "encoder.cl1.bias",
    "G_en/Conv_5/weights": "encoder.cl6.weight",
    "G_en/Conv_3/biases": "encoder.cl4.bias",
    "G_en/Conv/weights": "encoder.cl1.weight",
    "G_en/Conv_1/biases": "encoder.cl2.bias",
    "G_en/Conv_5/biases": "encoder.cl6.bias",
    "G_en/Conv_1/weights": "encoder.cl2.weight",
    "G_en/Conv_2/biases": "encoder.cl3.bias",
    "G_en/Conv_2/weights": "encoder.cl3.weight",
    "G_en/Conv_3/weights": "encoder.cl4.weight",
    "G_en/Conv_4/biases": "encoder.cl5.bias",
    "G_en/Conv_6/biases": "encoder.dcl1.bias",
    "G_en/Conv_4/weights": "encoder.cl5.weight",
    "G_en/Conv_6/weights": "encoder.dcl1.weight",
    "G_en/Conv_7/biases": "encoder.dcl2.bias",
    "G_en/Conv_7/weights": "encoder.dcl2.weight",
    "G_en/Conv_8/biases": "encoder.dcl3.bias",
    "G_en/Conv_8/weights": "encoder.dcl3.weight",
    "G_en/Conv_9/biases": "encoder.dcl4.bias",
    "G_en/Conv_9/weights": "encoder.dcl4.weight",
    "disc_red/l1b": "discriminator_red.l1.bias",
    "disc_red/l1w": "discriminator_red.l1.weight",
    "disc_red/l2b": "discriminator_red.l2.bias",
    "disc_red/l2w": "discriminator_red.l2.weight",
    "disc_red/l3b": "discriminator_red.l3.bias",
    "disc_red/l3w": "discriminator_red.l3.weight",
    "disc_red/l4b": "discriminator_red.l4.bias",
    "disc_red/l4w": "discriminator_red.l4.weight",
    "disc_red/l5b": "discriminator_red.l5.bias",
    "disc_red/l5w": "discriminator_red.l5.weight",
    "disc_red/l6b": "discriminator_red.l6.bias",
    "disc_red/l6w": "discriminator_red.l6.weight",
    "disc_red/l7_b": "discriminator_red.l7.bias",
    "disc_red/l7_w": "discriminator_red.l7.weight"
}
print("TensorFlow variables in checkpoint:")
k = 0
for key in var_to_shape_map:
    if "Adam" in key or "beta" in key or "u" in key:
        continue
    k+=1
    print(k, key, reader.get_tensor(key).shape)

# -------------------------------
# 2. Build your PyTorch model
# -------------------------------
import model_new# <-- replace with your model class
# import model
# print(f"tensorflow_model inpaint!!!!!!!!!")
# tensorflow_model = model.InpaintNN()
pytorch_model = model_new.InpaintNN()
# pytorch_model.eval()


# Print all parameter names and their shapes
k=0
for name, param in pytorch_model.named_parameters():
    k+=1
    print(f"{k}: Name: {name} | Shape: {param.shape}")

# exit()

# -------------------------------
# 3. Automatic name mapping
# -------------------------------
def convert_tf_to_pt_name(tf_name):
    # Basic replacements, adapt depending on your TF naming
    name = mapping.get(tf_name, None)
    return name


# -------------------------------
# 3. Create mapping dict
#    TF variable name -> PyTorch state_dict key
# -------------------------------
# name_mapping = {
#     # Encoder example
#     "G_en/conv1/w": "encoder.conv1.weight",
#     "G_en/conv1/b": "encoder.conv1.bias",

#     # Decoder example
#     "G_de/conv1/w": "decoder.conv1.weight",
#     "G_de/conv1/b": "decoder.conv1.bias",

#     # Discriminator example
#     "disc_red/conv1/w": "discriminator.conv1.weight",
#     "disc_red/conv1/b": "discriminator.conv1.bias",

#     # ... extend for all layers you need
# }


state_dict = pytorch_model.state_dict()

# -------------------------------
# 4. Transfer weights
# -------------------------------
for tf_name in var_to_shape_map:
    print("Processing TF variable:", tf_name)
    pt_name = convert_tf_to_pt_name(tf_name)
    if pt_name not in state_dict:
        # skip if the layer is not in PyTorch model
        print(f"Skipping {pt_name}")
        continue
    
    array = reader.get_tensor(tf_name)

    # Conv2D: H W in_ch out_ch -> out_ch in_ch H W
    if "weight" in pt_name and array.ndim == 4:
        array = array.transpose(3, 2, 0, 1)
    
    # Dense/Linear: in_features, out_features -> out_features, in_features
    elif "weight" in pt_name and array.ndim == 2:
        array = array.transpose(1, 0)
    
    tensor = torch.from_numpy(array)
    
    if tensor.shape != state_dict[pt_name].shape:
        print(f"Shape mismatch for {pt_name}: {tensor.shape} vs {state_dict[pt_name].shape}, skipping")
        continue
    
    state_dict[pt_name] = tensor

    print(f"Loaded {tf_name} → {pt_name} {tuple(tensor.shape)}")

# -------------------------------
# 5. Save converted model
# -------------------------------
pytorch_model.load_state_dict(state_dict)
torch.save(pytorch_model.state_dict(), "converted_model.pth")
print("✅ Conversion complete! Saved as converted_model.pth")
