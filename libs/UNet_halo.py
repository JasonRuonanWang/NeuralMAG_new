import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import copy
def initialize_model_with_pretrained_weights(new_model, pretrained_model_path):
    pretrained_weights = torch.load(pretrained_model_path)
    
    new_model_state_dict = new_model.state_dict()
    
    for name, param in new_model_state_dict.items():
        print(name)
        if name in pretrained_weights:
            new_model_state_dict[name].copy_(pretrained_weights[name])
        if name == 'weight_s0':
            new_model_state_dict[name].copy_(pretrained_weights['s0.weight'])
        if name == 'bias_s0':
            new_model_state_dict[name].copy_(pretrained_weights['s0.bias'])
        if name == 'weight_up0':
            new_model_state_dict[name].copy_(pretrained_weights['up0.up.1.weight'])
        if name == 'bias_up0':
            new_model_state_dict[name].copy_(pretrained_weights['up0.up.1.bias'])
        if name == 'weight_last_tr':
            new_model_state_dict[name].copy_(pretrained_weights['last_Conv.0.weight'])
        if name == 'bias_last_tr':
            new_model_state_dict[name].copy_(pretrained_weights['last_Conv.0.bias'])
        if name == 'weight_last_c':
            new_model_state_dict[name].copy_(pretrained_weights['last_Conv.3.weight'])
        if name == 'bias_last_c':
            new_model_state_dict[name].copy_(pretrained_weights['last_Conv.3.bias'])
        if 'norm' in name:
            if 'up0' in name:
                new_model_state_dict[name].copy_(pretrained_weights[name.replace('norm', 'up.2')])
            if 'last' in name:
                new_model_state_dict[name].copy_(pretrained_weights[name.replace('norm', '1')])
        if 'norm1' in name:
            new_model_state_dict[name].copy_(pretrained_weights[name.replace('norm1', 'up0.up.2')])
        if 'norm2' in name:
            new_model_state_dict[name].copy_(pretrained_weights[name.replace('norm2', 'last_Conv.1')])

    new_model.load_state_dict(new_model_state_dict)

class Slices:
    def __init__(
        self,
        device: torch.device,
    ):
        self.output = None
        self.avg = None
        self.var = None
        self.device = device
        self.neighbors: Dict[str, Optional['Slices']] = {
            'top': None,
            'bottom': None,
            'left': None,
            'right': None,
            'top_left': None,
            'top_right': None,
            'bottom_left': None,
            'bottom_right': None,
        }
    def set_neighbor(self, direction: str, neighbor: Optional['Slices']):
        assert direction in self.neighbors, f"Invalid direction: {direction}"
        self.neighbors[direction] = neighbor
    
    def get_average(self) -> torch.Tensor:
        avg = torch.mean(self.output, dim=(1, 2, 3), keepdim=True)
        self.avg = avg
        return avg
    
    def assign_average(self, avg: torch.Tensor):
        self.avg = avg.to(self.device)
        
    def get_variance(self) -> torch.Tensor:
        var = torch.mean(torch.square(self.output - self.avg), dim=(1, 2, 3), keepdim=True)
        return var

    def assign_variance(self, var: torch.Tensor):
        self.var = var.to(self.device)

    def normalize(self):
        self.output = (self.output - self.avg) / torch.sqrt(self.var + 1e-5)
    
    def self_normalize(self):
        avg = torch.mean(self.output, dim=(1, 2, 3), keepdim=True)
        var = torch.mean(torch.square(self.output - avg), dim=(1, 2, 3), keepdim=True)
        self.output = (self.output - avg) / torch.sqrt(var + 1e-5)

class DownsampleSlices(Slices):
    def __init__(
        self,
        device: torch.device,
        weight: Optional[torch.Tensor] = None,
        weight_0: Optional[torch.Tensor] = None,
        bias_0: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 1
    ):
        super(DownsampleSlices, self).__init__(device)
        self.output_list = []

        self.weight = weight.to(device) if weight is not None else None
        self.weight_0 = weight_0.to(device) if weight_0 is not None else None
        self.bias_0 = bias_0.to(device) if bias_0 is not None else None
        self.stride = stride
        self.padding = padding

        self.pool = nn.AvgPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
    

    def halo_exchange(self) -> torch.Tensor:
        padding = self.padding
        padded_input = F.pad(self.input, (padding, padding, padding, padding))

        if self.neighbors['top'] is not None:
            padded_input[:, :, :padding, padding:-padding] = \
                self.neighbors['top'].input[:, :, -padding:, :].to(self.device)
        if self.neighbors['bottom'] is not None:
            padded_input[:, :, -padding:, padding:-padding] = \
                self.neighbors['bottom'].input[:, :, :padding, :].to(self.device)
        if self.neighbors['left'] is not None:
            padded_input[:, :, padding:-padding, :padding] = \
                self.neighbors['left'].input[:, :, :, -padding:].to(self.device)
        if self.neighbors['right'] is not None:
            padded_input[:, :, padding:-padding, -padding:] = \
                self.neighbors['right'].input[:, :, :, :padding].to(self.device)

        if self.neighbors['top_left'] is not None:
            padded_input[:, :, :padding, :padding] = \
                self.neighbors['top_left'].input[:, :, -padding:, -padding:].to(self.device)
        if self.neighbors['top_right'] is not None:
            padded_input[:, :, :padding, -padding:] = \
                self.neighbors['top_right'].input[:, :, -padding:, :padding].to(self.device)
        if self.neighbors['bottom_left'] is not None:
            padded_input[:, :, -padding:, :padding] = \
                self.neighbors['bottom_left'].input[:, :, :padding, -padding:].to(self.device)
        if self.neighbors['bottom_right'] is not None:
            padded_input[:, :, -padding:, -padding:] = \
                self.neighbors['bottom_right'].input[:, :, :padding, :padding].to(self.device)

        return padded_input

    def conv(self, is_first: bool = False):
        padded_input = self.halo_exchange()
        if is_first:
            self.output = F.conv2d(padded_input, self.weight_0, bias=self.bias_0, stride = self.stride, padding = 0)
        else:
            self.output = F.conv2d(padded_input, self.weight, stride = self.stride, padding = 0)
    def assign_input(self, input: torch.Tensor):
        self.input = input.to(self.device)
        self.output = None
        self.avg = None
        self.output_list = [self.input]

    
    def synchronize(self):
        self.output_list.append(self.output)
        self.input = self.lrelu(self.pool(self.output)) if self.output.size(2) > 1 else None

class UpsampleSlices(Slices):
    def __init__(
        self,
        device: torch.device,
        weight: Optional[torch.Tensor] = None,
        weight_0: Optional[torch.Tensor] = None,
        bias_0: Optional[torch.Tensor] = None,
        weight_last_convtr: Optional[torch.Tensor] = None,
        bias_last_convtr: Optional[torch.Tensor] = None,
        weight_last_conv: Optional[torch.Tensor] = None,
        bias_last_conv: Optional[torch.Tensor] = None,
        stride: int = 2,
        padding: int = 1,
        ch1: int = 48,
        ch2: int = 6
    ):
        super(UpsampleSlices, self).__init__(device)

        self.weight = weight.to(device) if weight is not None else None
        self.weight_0 = weight_0.to(device) if weight_0 is not None else None
        self.bias_0 = bias_0.to(device) if bias_0 is not None else None
        self.weight_last_convtr = weight_last_convtr.to(device) if weight_last_convtr is not None else None
        self.bias_last_convtr = bias_last_convtr.to(device) if bias_last_convtr is not None else None
        self.weight_last_conv = weight_last_conv.to(device) if weight_last_conv is not None else None
        self.bias_last_conv = bias_last_conv.to(device) if bias_last_conv is not None else None
        self.stride = stride
        self.padding = padding

        self.relu = nn.ReLU(True)
        self.norm_0 = nn.BatchNorm2d(ch1)
        self.norm_last = nn.BatchNorm2d(ch2)
        self.tanh = nn.Tanh()
    
    def assign_input(self, input:torch.Tensor, input_list: List[torch.Tensor]):
        input_list = [x.to(self.device) for x in input_list]
        input = input.to(self.device)
        self.input = torch.cat([input,self.relu(input_list[-1])], dim=1)
        input_list.pop()
        self.dilate_input(stride=2)
        self.input_list = input_list
        self.output = None
        self.avg = None
        
    def dilate_input(self,stride=2):
        B, C, H, W = self.input.shape

        dilated_input = torch.zeros(B, C, 
                                    H * stride - stride + 1, 
                                    W * stride - stride + 1, 
                                    device=self.device)
        
        dilated_input[:, :, ::stride, ::stride] = self.input
        self.dilated_input = dilated_input

    def halo_exchange(self, kernel_size: int = 4 ) -> torch.Tensor:
        real_padding = kernel_size - self.padding - 1 
        padded_input = F.pad(self.dilated_input, (real_padding, real_padding, real_padding, real_padding))

        halo_size = 1
        if self.neighbors['top'] is not None:
            padded_input[:, :, :halo_size, real_padding:-real_padding] = \
                self.neighbors['top'].dilated_input[:, :, -halo_size:, :].to(self.device)
        if self.neighbors['bottom'] is not None:
            padded_input[:, :, -halo_size:, real_padding:-real_padding] = \
                self.neighbors['bottom'].dilated_input[:, :, :halo_size, :].to(self.device)
        if self.neighbors['left'] is not None:
            padded_input[:, :, real_padding:-real_padding, :halo_size] = \
                self.neighbors['left'].dilated_input[:, :, :, -halo_size:].to(self.device)
        if self.neighbors['right'] is not None:
            padded_input[:, :, real_padding:-real_padding, -halo_size:] = \
                self.neighbors['right'].dilated_input[:, :, :, :halo_size].to(self.device)
        if self.neighbors['top_left'] is not None:
            padded_input[:, :, :halo_size, :halo_size] = \
                self.neighbors['top_left'].input[:, :, -halo_size:, -halo_size:].to(self.device)
        if self.neighbors['top_right'] is not None:
            padded_input[:, :, :halo_size, -halo_size:] = \
                self.neighbors['top_right'].input[:, :, -halo_size:, :halo_size].to(self.device)
        if self.neighbors['bottom_left'] is not None:
            padded_input[:, :, -halo_size:, :halo_size] = \
                self.neighbors['bottom_left'].input[:, :, :halo_size, -halo_size:].to(self.device)
        if self.neighbors['bottom_right'] is not None:
            padded_input[:, :, -halo_size:, -halo_size:] = \
                self.neighbors['bottom_right'].input[:, :, :halo_size, :halo_size].to(self.device)

        return padded_input

    def conv(self):
        padded_input = self.halo_exchange()
        self.output = F.conv2d(padded_input, self.weight, stride=1, padding=0)
    
    def conv_0(self):
        padded_input = self.halo_exchange(kernel_size=3)
        self.output = F.conv2d(padded_input, self.weight_0, bias=self.bias_0, stride=1, padding=0)
        self.output = self.norm_0(self.output)

    def conv_last(self)->torch.Tensor:
        padded_input = self.halo_exchange(kernel_size=3)
        self.output = F.conv2d(padded_input, self.weight_last_convtr, bias=self.bias_last_convtr, stride=1, padding=0)
        norm_output = self.norm_last(self.output)
        tanh_input = self.tanh(norm_output)
        output = F.conv2d(tanh_input, self.weight_last_conv, bias=self.bias_last_conv, stride=1, padding=0)
        return output.to(torch.device('cuda:0'))


    def synchronize(self,stride=2): 
        self.output = self.relu(self.output)
        if len(self.input_list) > 1 or self.input_list[-1].size(2) > 64:
            self.input = torch.cat([self.output, self.relu(self.input_list[-1])], dim=1) 
            self.input_list.pop()
            self.dilate_input(stride) 

    def synchronize_last(self):
        self.input = torch.cat([self.output, self.input_list[-1]], dim=1)
        self.input_list.pop()
        self.dilate_input(stride=1)
class UNet(nn.Module):
    def __init__(self, kc, inc, ouc, device_list=[torch.device('cuda:0')], division_shape=(1,1)):
        super(UNet, self).__init__()
        self.device_list = device_list
        self.num_rows, self.num_cols = division_shape
        assert len(device_list) == self.num_rows * self.num_cols, "Device count must match division_shape"
        assert self.num_rows == self.num_cols , "Number of rows and columns must be equal"
        assert (self.num_rows & (self.num_rows - 1)) == 0, "Number of rows must be a power of 2"
        self.down_slices : List[DownsampleSlices] = []
        self.up_slices : List[UpsampleSlices] = []
        
        for device in device_list:
            self.down_slices.append(DownsampleSlices(device))
            self.up_slices.append(UpsampleSlices(device,ch1=kc, ch2=inc))
        self._assign_neighbors()

        self.down_slice_0 = DownsampleSlices(torch.device('cuda:0'))
        self.up_slices_0 = UpsampleSlices(torch.device('cuda:0'), ch1=kc, ch2=inc)

        self.weight_s0 = nn.Parameter(torch.zeros(1*kc, inc, 3, 3))
        self.bias_s0 = nn.Parameter(torch.zeros(1*kc))
        
        self.weight_s  = nn.Parameter(torch.zeros(1*kc, 1*kc, 3, 3))
        
        self.weight_up = nn.Parameter(torch.zeros(2*kc, 1*kc, 4, 4))

        self.weight_up0 = nn.Parameter(torch.zeros(2*kc, 1*kc, 3, 3))
        self.bias_up0 = nn.Parameter(torch.zeros(1*kc))
        self.norm1 = nn.BatchNorm2d(kc)
        
        self.weight_last_tr = nn.Parameter(torch.zeros(kc+inc, 1*kc, 3, 3))
        self.bias_last_tr = nn.Parameter(torch.zeros(1*kc))
        self.weight_last_c = nn.Parameter(torch.zeros(ouc, 1*kc, 1, 1))
        self.bias_last_c = nn.Parameter(torch.zeros(ouc))
        self.norm2 = nn.BatchNorm2d(kc)
    def _assign_neighbors(self):
        for idx in range(len(self.device_list)):
            row = idx // self.num_cols
            col = idx % self.num_cols

            self.down_slices[idx].set_neighbor('top', self._get_neighbor(row-1, col, is_down=True))
            self.down_slices[idx].set_neighbor('bottom', self._get_neighbor(row+1, col, is_down=True))
            self.down_slices[idx].set_neighbor('left', self._get_neighbor(row, col-1, is_down=True))
            self.down_slices[idx].set_neighbor('right', self._get_neighbor(row, col+1, is_down=True))
            self.down_slices[idx].set_neighbor('top_left', self._get_neighbor(row-1, col-1, is_down=True))
            self.down_slices[idx].set_neighbor('top_right', self._get_neighbor(row-1, col+1, is_down=True))
            self.down_slices[idx].set_neighbor('bottom_left', self._get_neighbor(row+1, col-1, is_down=True))
            self.down_slices[idx].set_neighbor('bottom_right', self._get_neighbor(row+1, col+1, is_down=True))
            
            self.up_slices[idx].set_neighbor('top', self._get_neighbor(row-1, col, is_down=False))
            self.up_slices[idx].set_neighbor('bottom', self._get_neighbor(row+1, col, is_down=False))
            self.up_slices[idx].set_neighbor('left', self._get_neighbor(row, col-1, is_down=False))      
            self.up_slices[idx].set_neighbor('right', self._get_neighbor(row, col+1, is_down=False))
            self.up_slices[idx].set_neighbor('top_left', self._get_neighbor(row-1, col-1, is_down=False))
            self.up_slices[idx].set_neighbor('top_right', self._get_neighbor(row-1, col+1, is_down=False))
            self.up_slices[idx].set_neighbor('bottom_left', self._get_neighbor(row+1, col-1, is_down=False))
            self.up_slices[idx].set_neighbor('bottom_right', self._get_neighbor(row+1, col+1, is_down=False))

    def _get_neighbor(self, row: int, col: int, is_down: bool) -> Optional[Slices]:
        if 0 <= row < self.num_rows and 0 <= col < self.num_cols:
            idx = row * self.num_cols + col
            return self.down_slices[idx] if is_down else self.up_slices[idx]
        return None
    
    def load_weights(self,cpkt):
        initialize_model_with_pretrained_weights(self, cpkt)
        self.down_slice_0.weight = self.weight_s.to(self.down_slice_0.device)
        for down_slice in self.down_slices:
            down_slice.weight = self.weight_s.to(down_slice.device)
            down_slice.weight_0 = self.weight_s0.to(down_slice.device)
            down_slice.bias_0 = self.bias_s0.to(down_slice.device)

        weight_up = self.weight_up.flip(2).flip(3).transpose(1, 0)
        weight_up0 = self.weight_up0.flip(2).flip(3).transpose(1, 0)
        weight_last_tr = self.weight_last_tr.flip(2).flip(3).transpose(1, 0)
        
        self.up_slices_0.weight = weight_up.to(self.up_slices_0.device)
        for up_slice in self.up_slices:
            up_slice.weight = weight_up.to(up_slice.device)
            up_slice.weight_0 = weight_up0.to(up_slice.device)
            up_slice.bias_0 = self.bias_up0.to(up_slice.device)
            up_slice.weight_last_convtr = weight_last_tr.to(up_slice.device)
            up_slice.bias_last_convtr = self.bias_last_tr.to(up_slice.device)
            up_slice.weight_last_conv = self.weight_last_c.to(up_slice.device)
            up_slice.bias_last_conv = self.bias_last_c.to(up_slice.device)
            up_slice.norm_0 =  copy.deepcopy(self.norm1).to(up_slice.device)
            up_slice.norm_last = copy.deepcopy(self.norm2).to(up_slice.device)
    def downsample_step(self, is_first: bool = False):
        avg = 0
        var = 0
        if is_first:
            for down_slice in self.down_slices:
                down_slice.conv(is_first)
            for down_slice in self.down_slices:
                down_slice.synchronize()
        else:
            for down_slice in self.down_slices:
                down_slice.conv()
                avg += down_slice.get_average().to(torch.device('cpu'))
            avg /= len(self.device_list)

            for down_slice in self.down_slices:
                down_slice.assign_average(avg)
                var += down_slice.get_variance().to(torch.device('cpu'))
            var /= len(self.device_list)

            for down_slice in self.down_slices:
                down_slice.assign_variance(var)
                down_slice.normalize()
            for down_slice in self.down_slices:
                down_slice.synchronize()

    def upsample_step(self,stride=2):
        avg = 0
        var = 0
        for up_slice in self.up_slices:
            up_slice.conv()
            avg += up_slice.get_average().to(torch.device('cpu'))
        avg /= len(self.device_list)
        for up_slice in self.up_slices:
            up_slice.assign_average(avg)
            var += up_slice.get_variance().to(torch.device('cpu'))
        var /= len(self.device_list)
        for up_slice in self.up_slices:
            up_slice.assign_variance(var)
            up_slice.normalize()
        for up_slice in self.up_slices:
            up_slice.synchronize(stride)
    
    def forward(self, x):
        size = x.size(2)
        output = torch.zeros_like(x, device=torch.device('cuda:0'))
        layer_num = 0
        while (1 << layer_num) < size:
            layer_num += 1
        assert layer_num>=8, "Input size must be at least 256x256"
        num_rows = self.num_rows
        num_cols = self.num_cols

        for idx, device in enumerate(self.device_list):
            i = idx // self.num_cols
            j = idx % self.num_cols
            input_slice = x[:, :, 
                            i*size//num_rows:(i+1)*size//num_rows, 
                            j*size//num_cols:(j+1)*size//num_cols].to(device)
            self.down_slices[idx].assign_input(input_slice)

        for i in range(layer_num):
            if i == 0:
                self.downsample_step(is_first=True)
            self.downsample_step()
            B,C,H,W = self.down_slices[0].input.shape # H=W
            if H*num_rows <= 64:
                break
        inter_input = torch.zeros((B
                                   , C, H*num_rows, H*num_rows), device='cuda:0')
        for idx, down_slice in enumerate(self.down_slices):
            i = idx // self.num_cols
            j = idx % self.num_cols
            input_slice = down_slice.input
            inter_input[:, :, i*H:(i+1)*H, j*W:(j+1)*W] = input_slice

        
        self.down_slice_0.assign_input(inter_input)
        for i in range(7):
            self.down_slice_0.conv()
            self.down_slice_0.self_normalize()
            self.down_slice_0.synchronize()

        self.up_slices_0.assign_input(self.down_slice_0.output, self.down_slice_0.output_list)
        for i in range(7):
            self.up_slices_0.conv()
            self.up_slices_0.self_normalize()
            self.up_slices_0.synchronize()

        B,C,H,W = self.up_slices_0.output.shape
        for idx, (down_slice, up_slice) in enumerate(zip(self.down_slices, self.up_slices)):
            i = idx // self.num_cols
            j = idx % self.num_cols
            input_slice = self.up_slices_0.output[:, :, 
                                                 i*H//num_rows:(i+1)*H//num_rows, 
                                                 j*W//num_cols:(j+1)*W//num_cols]
            up_slice.assign_input(input_slice, down_slice.output_list)

        for i in range(layer_num-8):
            self.upsample_step()

        self.upsample_step(stride=1)
        for up_slice in self.up_slices:
            up_slice.conv_0()
        for up_slice in self.up_slices:
            up_slice.synchronize_last()
        for up_slice in self.up_slices:
            up_slice.conv_last()
        
        for idx, up_slice in enumerate(self.up_slices):
            i = idx // self.num_cols
            j = idx % self.num_cols
            output_slice = up_slice.conv_last()
            output[:, :,
                   i*size//num_rows:(i+1)*size//num_rows, 
                   j*size//num_cols:(j+1)*size//num_cols] = output_slice
            
        return torch.where(output >= 0, torch.exp(output)-1, -torch.exp(-output)+1)
        # return output

