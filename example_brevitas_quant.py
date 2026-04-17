import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from sapfa import run_fault_injection
from sapfa.quantization import QuantConfig, quantize

if __name__ == '__main__':
    
    # Set the device
    if torch.cuda.is_available():
        my_device = torch.device('cuda:0')
    else:
        my_device = torch.device('cpu')
            
    # Initialize the network (VGG-11 for CIFAR-10)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar10_vgg11_bn', pretrained=True)
    model.to(my_device)
    
    # Create calibration dataloader
    calib_batch_size = 16
    calib_samples = 5000
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
    calib_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
    calib_sampler = torch.utils.data.RandomSampler(calib_dataset, num_samples=calib_samples)
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=calib_batch_size, sampler=calib_sampler, num_workers=1)
    
    # Create validation and testing dataloader
    val_batch_size = 16
    val_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)
    
    # Init default config for quantization
    config = QuantConfig()
    
    # Quantize model
    model = quantize(model, config, calib_loader, val_loader, my_device)
    
    # Run fault injection (set int_ops to False)
    run_fault_injection(model, val_loader, my_device, sa_size=32, fault_num=10, dataflow='OS', 
                        fault_type='permanent', quant=True, int_ops=False, stuck_at=1)
