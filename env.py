import torch
import torch_cluster
import pytorch3d

print(f"✅ PyTorch3D 版本: {pytorch3d.__version__}")
print(f"✅ TorchCluster 版本: {torch_cluster.__version__}")
print(f"✅ 当前显卡: {torch.cuda.get_device_name(0)}")
print(f"✅ 算力版本: {torch.cuda.get_device_capability(0)}") # 4080 应显示 (8, 9)