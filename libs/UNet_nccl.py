import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, Dict, List
import copy
def initialize_model_with_pretrained_weights(new_model, pretrained_model_path):
    pretrained_weights = torch.load(pretrained_model_path)
    
    new_model_state_dict = new_model.state_dict()
    
    for name, param in new_model_state_dict.items():
        print(name)
        if name in pretrained_weights:
            new_model_state_dict[name].copy_(pretrained_weights[name])
            continue
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

def scatter_slices(rank: int, x: torch.Tensor, split: int, device_list: List[torch.device]):
    if rank == 0:
        batch_size, channels, size, _ = x.size()
        split_size = size // split
        slice_list = []
        shape_tensor = torch.tensor((batch_size, channels, split_size), device=device_list[rank], dtype=torch.int)
        dist.broadcast(shape_tensor, src=0)
    else:
        shape_tensor = torch.zeros(3, device=device_list[rank], dtype=torch.int)
        dist.broadcast(shape_tensor, src=0)
        batch_size, channels, split_size = shape_tensor.tolist()
        slice_list = None
    if rank == 0:
        for i in range(split):
            for j in range(split):
                slice = x[:, :, 
                            i*split_size:(i+1)*split_size, 
                            j*split_size:(j+1)*split_size].contiguous()
                slice_list.append(slice)
    local_slice = torch.empty((batch_size, channels, split_size, split_size), device=device_list[rank])
    dist.scatter(local_slice, scatter_list=slice_list, src=0)
    return local_slice

def gather_slices(rank, local_slice: torch.Tensor, split: int, device_list: List[torch.device]):
    batch_size, channels, split_size, _ = local_slice.size()
    size = split_size * split
    if rank == 0:
        gathered_slices = [torch.empty((batch_size, channels, split_size, split_size), 
                                    device=device_list[rank]) 
                                    for i in range(len(device_list))]
        dist.barrier()
        dist.gather(local_slice, gather_list=gathered_slices, dst=0)
        output = torch.empty((batch_size, channels, size, size), device=device_list[rank])
        for i in range(split):
            for j in range(split):
                output[:, :, 
                       i*split_size:(i+1)*split_size, 
                       j*split_size:(j+1)*split_size] = gathered_slices[i*split + j]
        return output
    else:
        dist.barrier()
        dist.gather(local_slice, gather_list=None, dst=0)
        return None      

class Slices:
    def __init__(
        self,
        device: torch.device,
    ):
        self.row = None
        self.col = None
        self.split = None

        self.output = None
        self.avg = None
        self.var = None
        self.device = device
        self.world_size = None
    def set_neighbor(self, direction: str, neighbor: Optional['Slices']):
        assert direction in self.neighbors, f"Invalid direction: {direction}"
        self.neighbors[direction] = neighbor
    
    def get_average(self) -> torch.Tensor:
        avg = torch.mean(self.output, dim=(1, 2, 3), keepdim=True)
        self.avg = avg
        return avg
    
    def get_variance(self) -> torch.Tensor:
        var = torch.mean(torch.square(self.output - self.avg), dim=(1, 2, 3), keepdim=True)
        return var

    def dist_normalize(self, world_size: int):
        avg = self.get_average()
        dist.all_reduce(avg, op=dist.ReduceOp.SUM)
        avg /= world_size

        var = self.get_variance()
        dist.all_reduce(var, op=dist.ReduceOp.SUM)
        var /= world_size

        self.output = (self.output - avg) / torch.sqrt(var + 1e-5)

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
        input = self.input
        padded_input = F.pad(input, (padding, padding, padding, padding))

        split = self.split
        if split == 1:
            return padded_input
        
        row = self.row
        col = self.col
        B, C, size, _ = input.size()
        # Create a halo tensor that contains the necessary padding for halo exchange
        halo_size = padding # 1
        halo_tensor = torch.cat([input[:, :, 0, :],
                                 input[:, :, :, 0],
                                 input[:, :, :, -1],
                                 input[:, :, -1, :]]
                                 ,dim=2
                                 ).contiguous()
        halo_list = [halo_tensor.clone() for _ in range(self.world_size)]
        dist.all_to_all(halo_list, halo_list)
        if row > 0:
            if col > 0:
                padded_input[:, :, :1, :1] = \
                    halo_list[(row - 1) * split + (col - 1)][:, :, (size*4-1):size*4].reshape(B, C, 1, 1) # top_left
            padded_input[:, :, :1, padding:-padding] = \
                halo_list[(row - 1) * split + col][:, :, size*3:size*4].reshape(B, C, 1, size) # top
            if col  < self.split - 1:
                padded_input[:, :, :1, -1:] = \
                    halo_list[(row - 1) * split + (col + 1)][:, :, size*3:(size*3+1)].reshape(B, C, 1, 1) # top_right
        if col > 0:
            padded_input[:, :, padding:-padding, :1] = \
                halo_list[row * split + (col - 1)][:, :, size*2:size*3].reshape(B, C, size, 1) # left
        if col < self.split - 1:
            padded_input[:, :, padding:-padding, -1:] = \
                halo_list[row * split + (col + 1)][:, :, size:size*2].reshape(B, C, size, 1) # right
        if row < self.split - 1:
            if col > 0:
                padded_input[:, :, -1:, :1] = \
                    halo_list[(row + 1) * split + (col - 1)][:, :, (size-1):size].reshape(B, C, 1, 1) # bottom_left
            padded_input[:, :, -1:, padding:-padding] = \
                halo_list[(row + 1) * split + col][:, :, :size].reshape(B, C, 1, size) # bottom 
            if col < self.split - 1:      
                padded_input[:, :, -1:, -1:] = \
                    halo_list[(row + 1) * split + (col + 1)][:, :, :1].reshape(B, C, 1, 1) # bottom_right
        
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
        real_padding = kernel_size - 2
        padded_input = F.pad(self.dilated_input, (real_padding, real_padding, real_padding, real_padding))
        split = self.split
        if split == 1:
            return padded_input
        row = self.row
        col = self.col
        B, C, size, _ = self.dilated_input.size()
        # Create a halo tensor that contains the necessary padding for halo exchange
        halo_size = 1
        halo_tensor = torch.cat([self.dilated_input[:, :, :1, :].flatten(2),
                                       self.dilated_input[:, :, : , 1:].flatten(2),
                                       self.dilated_input[:, :, :, :-1].flatten(2),
                                       self.dilated_input[:, :, -1, :].flatten(2)]
                                       ,dim=2
                                       )
        halo_list = [halo_tensor] * self.world_size
        dist.all_to_all(halo_list, halo_list)
        # top_left = halo_tensor[:, :, :1].reshape(B, C, 1, 1)
        # top = halo_tensor[:, :, :size].reshape(B, C, 1, size-2)
        # top_right = halo_tensor[:, :, (size-1):size].reshape(B, C, 1, 1)
        # left = halo_tensor[:, :, size:size*2].reshape(B, C, size-2, 1)
        # right = halo_tensor[:, :, size*2:size*3].reshape(B, C, size-2, 1)
        # bottom_left = halo_tensor[:, :, size*3:(size*3+1)].reshape(B, C, 1, 1)
        # bottom = halo_tensor[:, :, size*3:size*4].reshape(B, C, 1, size-2)
        # bottom_right = halo_tensor[:, :, (size*4-1):size*4].reshape(B, C, 1, 1)

        if row > 0:
            if col > 0:
                padded_input[:, :, :1, :1] = \
                    halo_list[(row - 1) * split + (col - 1)][:, :, (size*4-1):size*4].reshape(B, C, 1, 1) # top_left
            padded_input[:, :, :1, real_padding:-real_padding] = \
                halo_list[(row - 1) * split + col][:, :, size*3:size*4].reshape(B, C, 1, size) # top
            if col  < self.split - 1:
                padded_input[:, :, :1, -1:] = \
                    halo_list[(row - 1) * split + (col + 1)][:, :, size*3:(size*3+1)].reshape(B, C, 1, 1) # top_right
        if col > 0:
            padded_input[:, :, real_padding:-real_padding, :1] = \
                halo_list[row * split + (col - 1)][:, :, size*2:size*3].reshape(B, C, size, 1) # left
        if col < self.split - 1:
            padded_input[:, :, real_padding:-real_padding, -1:] = \
                halo_list[row * split + (col + 1)][:, :, size:size*2].reshape(B, C, size, 1) # right
        if row < self.split - 1:
            if col > 0:
                padded_input[:, :, -1:, :1] = \
                    halo_list[(row + 1) * split + (col - 1)][:, :, (size-1):size].reshape(B, C, 1, 1) # bottom_left
            padded_input[:, :, -1:, real_padding:-real_padding] = \
                halo_list[(row + 1) * split + col][:, :, :size].reshape(B, C, 1, size) # bottom 
            if col < self.split - 1:      
                padded_input[:, :, -1:, -1:] = \
                    halo_list[(row + 1) * split + (col + 1)][:, :, :1].reshape(B, C, 1, 1) # bottom_right
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
        return output


    def synchronize(self,stride=2): 
        self.output = self.relu(self.output)

        if len(self.input_list) > 1 or self.input_list[-1].size(2) > 64:
            self.input = torch.cat([self.output, self.relu(self.input_list[-1])], dim=1) 
            self.input_list.pop()
            self.dilate_input(stride) 

    def synchronize_last(self):
        self.input = torch.cat([self.output, self.input_list[-1]], dim=1)
        # self.input_list.pop()
        self.dilate_input(stride=1)
    
class UNet(nn.Module):
    def __init__(self, kc, inc, ouc, device_list , split):
        super(UNet, self).__init__()
        self.device_list = device_list
        self.split = split
        self.world_size = len(device_list)
        assert len(device_list) >= split * split, "Device count must match split"
        assert (split & (split - 1)) == 0, "split must be a power of 2"
        self.down_slices : List[DownsampleSlices] = []
        self.up_slices : List[UpsampleSlices] = []
        
        for device in device_list:
            self.down_slices.append(DownsampleSlices(device))
            self.up_slices.append(UpsampleSlices(device,ch1=kc, ch2=inc))
  
        self.down_slice_0 = DownsampleSlices(device_list[0])
        self.up_slices_0 = UpsampleSlices(device_list[0], ch1=kc, ch2=inc)
        self._assign_location()

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
    def _assign_location(self):
        for idx in range(len(self.device_list)):
            row = idx // self.split
            col = idx % self.split

            self.down_slices[idx].row = row
            self.down_slices[idx].col = col
            self.down_slices[idx].split = self.split
            self.down_slices[idx].world_size = len(self.device_list)
            self.up_slices[idx].row = row
            self.up_slices[idx].col = col
            self.up_slices[idx].split = self.split
            self.up_slices[idx].world_size = len(self.device_list)
            
        self.down_slice_0.split = 1
        self.up_slices_0.split = 1
    
    def load_weights(self,cpkt):
        initialize_model_with_pretrained_weights(self, cpkt)
        self.down_slice_0.weight = self.weight_s.to(self.down_slice_0.device).detach().clone()
        for down_slice in self.down_slices:
            down_slice.weight = self.weight_s.to(down_slice.device).detach().clone()
            down_slice.weight_0 = self.weight_s0.to(down_slice.device).detach().clone()
            down_slice.bias_0 = self.bias_s0.to(down_slice.device).detach().clone()

        weight_up = self.weight_up.flip(2).flip(3).transpose(1, 0)
        weight_up0 = self.weight_up0.flip(2).flip(3).transpose(1, 0)
        weight_last_tr = self.weight_last_tr.flip(2).flip(3).transpose(1, 0)
        
        self.up_slices_0.weight = weight_up.to(self.up_slices_0.device).detach().clone()
        for up_slice in self.up_slices:
            up_slice.weight = weight_up.to(up_slice.device).detach().clone()
            up_slice.weight_0 = weight_up0.to(up_slice.device).detach().clone()
            up_slice.bias_0 = self.bias_up0.to(up_slice.device).detach().clone()
            up_slice.weight_last_convtr = weight_last_tr.to(up_slice.device).detach().clone()
            up_slice.bias_last_convtr = self.bias_last_tr.to(up_slice.device).detach().clone()
            up_slice.weight_last_conv = self.weight_last_c.to(up_slice.device).detach().clone()
            up_slice.bias_last_conv = self.bias_last_c.to(up_slice.device).detach().clone()
            up_slice.norm_0 =  copy.deepcopy(self.norm1).to(up_slice.device)
            up_slice.norm_last = copy.deepcopy(self.norm2).to(up_slice.device)
            up_slice.norm_0.eval()
            up_slice.norm_last.eval()
    
    def _setup(self, rank: int, world_size: int):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    def _cleanup(self):
        dist.destroy_process_group()
        torch.cuda.empty_cache()
    def _unet_dist(self, rank: int, input: torch.Tensor):
        with torch.no_grad():
            world_size = self.world_size
            self._setup(rank, world_size)
            down_slice = self.down_slices[rank]
            up_slice = self.up_slices[rank]
            split = self.split

            size = input.size(2)
            layer_num = 0
            while (1 << layer_num) < size:
                layer_num += 1
            local_input = scatter_slices(rank, input, split, self.device_list)
            down_slice.assign_input(local_input)
            down_slice.conv(is_first=True)
            down_slice.synchronize()
            for i in range(layer_num):
                down_slice.conv()
                down_slice.dist_normalize(world_size)
                down_slice.synchronize()
                temp_size = down_slice.input.size(2)
                if temp_size*split <= 64 or temp_size <= 4:
                    inter_layer_num = 0
                    while (1 << inter_layer_num) < temp_size*split:
                        inter_layer_num += 1
                    break
            inter_input = gather_slices(rank, down_slice.input, split, self.device_list)
            if rank == 0:
                self.down_slice_0.assign_input(inter_input)
                for i in range(inter_layer_num + 1):
                    self.down_slice_0.conv()
                    self.down_slice_0.self_normalize()
                    self.down_slice_0.synchronize()

                self.up_slices_0.assign_input(self.down_slice_0.output, self.down_slice_0.output_list)
                for i in range(inter_layer_num + 2):
                    self.up_slices_0.conv()
                    self.up_slices_0.self_normalize()
                    self.up_slices_0.synchronize()

            inter_output = scatter_slices(rank, self.up_slices_0.output, split, self.device_list)
            up_slice.assign_input(inter_output, down_slice.output_list)

            for i in range(layer_num - inter_layer_num - 1):
                up_slice.conv()
                up_slice.dist_normalize(world_size)
                if i == layer_num - inter_layer_num - 2:
                    up_slice.synchronize(stride=1)
                else:
                    up_slice.synchronize()

            up_slice.conv_0()
            up_slice.synchronize_last()
            last_conv_output = up_slice.conv_last()
            output = gather_slices(rank, last_conv_output, split, self.device_list)

            if rank == 0:
                output = torch.where(output >= 0, torch.exp(output)-1, -torch.exp(-output)+1)
                torch.save(output, "output/output.pt")
            dist.barrier()
            self._cleanup() 

    def forward(self, x):        
        x = x.to(self.device_list[0]).detach()
        
        if os.path.exists("output/output.pt"):
            os.remove("output/output.pt")

        mp.spawn(
            self._unet_dist,
            args=(x,),
            nprocs=len(self.device_list),
            join=True
        )
        if not os.path.exists("output/output.pt"):
            raise RuntimeError("Output file not found. Ensure the distributed setup is correct.")
        output = torch.load("output/output.pt", map_location=self.device_list[0])
        return output
    
    