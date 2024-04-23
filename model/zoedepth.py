import torch

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo

ZoeDepthEstimator = torch.hub.load("isl-org/ZoeDepth", "ZoeD_K", pretrained=True)