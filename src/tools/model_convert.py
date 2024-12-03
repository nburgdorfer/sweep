import torch
import sys

filename="./pretrained_models/GBiNet/DTU/model_orig.pt"
new_filename="./pretrained_models/GBiNet/DTU/model.pt"

model = torch.load(filename)["model"]
new_model = {}
for k,v in model.items():
    k_list = k.split('.')
    if k_list[3] == "norm_layer":
        k_list[3] = "norm"
        k = ".".join(k_list)

    if k[:11] == "img_feature":
        if k[12:24] == "conv_inner.1":
            k = f"feature_encoder.conv4.conv{k[24:]}"
        elif k[12:24] == "conv_inner.2":
            k = f"feature_encoder.conv5.conv{k[24:]}"
        elif k[12:24] == "conv_inner.3":
            k = f"feature_encoder.conv6.conv{k[24:]}"
        elif k[12:22] == "conv_out.0": 
            k = f"feature_encoder.out0{k[22:]}"
        elif k[12:22] == "conv_out.1": 
            k = f"feature_encoder.out1{k[22:]}"
        elif k[12:22] == "conv_out.2": 
            k = f"feature_encoder.out2{k[22:]}"
        elif k[12:22] == "conv_out.3": 
            k = f"feature_encoder.out3{k[22:]}"
        else:
            k = f"feature_encoder{k[11:]}"
    if k[:12] == "cost_network":
        k = f"cost_reg{k[12:]}"
    new_model[k] = v

print(new_model.keys())
torch.save({"model": new_model}, new_filename)
