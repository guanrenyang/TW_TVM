import torch
import torchvision.models as models

# Initialize pre-trained ResNet18 model
model = models.alexnet(pretrained=True)

# Iterate through each layer in the model
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        # Calculate L2 norm of weight tensor
        l2_norm = torch.norm(module.weight.data, p=2)
        # Count number of parameters in weight tensor
        num_params = module.weight.data.numel()
        # Print layer name, L2 norm, and number of parameters
        print(f"{name}: L2 norm: {l2_norm:.4f} \t Number of parameters: {num_params} \t {l2_norm*1e8/num_params:.4f}")

