import torch
import torch.nn as nn
import torchvision

from safpa import run_fault_injection


class QAlexNet(nn.Module):
    """Fully quantized Alexnet model for CIFAR-10 dataset."""
    def __init__(self, bit_width=8):
        super().__init__()
        self.bit_width = bit_width
        
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, 256)
        self.fc8 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()
           
    def forward(self, images):
        x = self.rescale(self.relu(self.conv1(images)))
        x = self.pool1(x)
        x = self.rescale(self.relu(self.conv2(x)))
        x = self.pool2(x)
        x = self.rescale(self.relu(self.conv3(x)))
        x = self.rescale(self.relu(self.conv4(x)))
        x = self.pool4(x)
        x = self.rescale(self.relu(self.conv5(x)))
        x = self.pool5(x)
        x = torch.flatten(x.permute(0, 2, 3, 1), 1)
        x = self.rescale(self.relu(self.fc6(x)))
        x = self.rescale(self.relu(self.fc7(x)))
        x = self.rescale(self.fc8(x))
        
        return x
    
    def rescale(self, in_activation):
        """Rescale output value to bit_width."""
        
        size = in_activation.size()
        batch = size[0]
        if in_activation.dim() == 4:
            features = size[1] * size[2] * size[3]
        else:
            features = size[1]
        
        in_act_cp = torch.reshape(in_activation, (batch, features))
        
        min_act, _ = torch.min(in_act_cp, 1)
        max_act, _ = torch.max(in_act_cp, 1)
        
        scaling_factor = (max_act - min_act) / (2 ** self.bit_width - 1)
        scaling_factor = torch.unsqueeze(scaling_factor, 1)
        
        out_activation = torch.round(in_act_cp / (scaling_factor))
        out_activation = torch.nan_to_num(out_activation)
        out_activation = torch.reshape(out_activation, in_activation.size())
        
        return out_activation

class ToTensor(object):
    """Convert PIL image to Tensor without rescaling."""

    def __call__(self, image):
        image = torchvision.transforms.functional.pil_to_tensor(image)
        return image.to(dtype=torch.float32) - 122


if __name__ == '__main__':
    
    # Set the device
    if torch.cuda.is_available():
        my_device = torch.device('cuda:0')
    else:
        my_device = torch.device('cpu')
    
    # Initialize the network
    model = QAlexNet()
    model.load_state_dict(torch.load('models/alexnet_qint8.pt', weights_only=True))
    model.to(my_device)
    
    # Create dataloader
    batch_size = 16
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # Run fault injection (set int_ops to True)
    run_fault_injection(model, dataloader, my_device, sa_size=32, fault_num=10, dataflow='OS', 
                        fault_type='transient', quant=True, int_ops=True)