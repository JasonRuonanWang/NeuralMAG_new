import torch
import torch.nn as nn
import torch.nn.functional as F   
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
        
    new_model.load_state_dict(new_model_state_dict)

def manual_conv2d_multi_gpu(input, weight, bias, stride, padding, device_list, division_shape, stream_list):
    batch, _, size, _ = input.size()
    out_ch, _, k_size, _ = weight.size()

    num_gpus = len(device_list)
    if size < 256:
        num_gpus = 1
        division_shape = (1, 1)
    num_h, num_w = division_shape

    padded_input = F.pad(input, (padding, padding, padding, padding))

    weight_list = [weight.to(device) for device in device_list]
    output = torch.zeros(batch, out_ch, size, size, dtype=input.dtype, device=torch.device('cuda:0'))
    
    if bias is not None:
        bias = bias.to(torch.device('cuda:0'))

    for index in range(num_gpus):
        i = index // num_w
        j = index % num_w
        input_slice = padded_input[:, :, 
                        i*size//num_h:(i+1)*size//num_h + k_size -1, 
                        j*size//num_w:(j+1)*size//num_w + k_size -1].to(device_list[index])
        output_slice = F.conv2d(
            input_slice, 
            weight_list[index], 
            stride=stride, 
            padding=0
        )
        
        output[:,:,
            i*size//num_h:(i+1)*size//num_h,
            j*size//num_w:(j+1)*size//num_w
            ] = output_slice.to(torch.device('cuda:0'))

    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output

def manual_convtranspose2d_multi_gpu(input, weight, bias, stride, padding, device_list, division_shape, stream_list):
    batch, _, size, _ = input.size()
    batch, ch, size, _ = input.size()
    _, out_ch, k_size, _ = weight.size()
    tr_weight = weight.flip(2).flip(3).transpose(1, 0)

    num_gpus = len(device_list)
    if size < 256:
        num_gpus = 1
        division_shape = (1, 1)
    num_h, num_w = division_shape

    dilated_input = torch.zeros(batch, ch, size * stride - stride + 1, size * stride - stride + 1,
                                dtype=input.dtype, device=input.device)
    dilated_input[:, :, ::stride, ::stride] = input
    real_padding = k_size - padding - 1 
    padded_input = F.pad(dilated_input, (real_padding, real_padding, real_padding, real_padding))

    tr_weight_list = [tr_weight.to(device) for device in device_list]
    output = torch.zeros(batch, out_ch, size * stride, size * stride, dtype=input.dtype, device=torch.device('cuda:0'))
    
    if bias is not None:
        bias = bias.to(torch.device('cuda:0'))

    for index in range(num_gpus):
        i = index // num_w
        j = index % num_w
        input_slice = padded_input[:, :,
                        i*stride*size//num_h:(i+1)*stride*size//num_h + k_size - 1,
                        j*stride*size//num_w:(j+1)*stride*size//num_w + k_size - 1].to(device_list[index])
        output_slice = F.conv2d(
            input_slice, 
            tr_weight_list[index],
            stride=1, 
            padding=0
        )

        output[:,:,
            i*stride*size//num_h:(i+1)*stride*size//num_h,
            j*stride*size//num_w:(j+1)*stride*size//num_w
            ] = output_slice.to(torch.device('cuda:0'))


    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output

class downsample(nn.Module):
    def __init__(self, out_channels, s, p, device_list, division_shape, stream_list):
        super(downsample, self).__init__()
        self.stride = s
        self.padding = p
        self.device_list = device_list
        self.division_shape = division_shape
        self.stream_list = stream_list

        self.pool = nn.AvgPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.norm = nn.GroupNorm(1, out_channels, affine=False)

    def forward(self, x, weight):
        pool_x = self.pool(x)
        lrelu_x = self.lrelu(pool_x)
        conv_x = manual_conv2d_multi_gpu(
            lrelu_x, 
            weight, 
            bias = None, 
            stride=self.stride, 
            padding=self.padding, 
            device_list=self.device_list, 
            division_shape=self.division_shape,
            stream_list=self.stream_list
        )
        norm_x = self.norm(conv_x)
        return norm_x
    
class upsample(nn.Module):
    def __init__(self, out_channels, s, p, device_list, division_shape, stream_list):
        super(upsample, self).__init__()
        self.stride = s
        self.padding = p
        self.device_list = device_list
        self.division_shape = division_shape
        self.stream_list = stream_list

        self.relu = nn.ReLU(True)
        self.norm = nn.GroupNorm(1, out_channels, affine=False)
        
    def forward(self, x, weight):
        relu_x = self.relu(x)
        convt_x = manual_convtranspose2d_multi_gpu(
            relu_x, 
            weight, 
            bias = None, 
            stride=self.stride,
            padding=self.padding, 
            device_list=self.device_list, 
            division_shape=self.division_shape,
            stream_list=self.stream_list
        )
        norm_x = self.norm(convt_x)
        return norm_x
        
class upsample_0(nn.Module):
    def __init__(self, out_channels, s, p, device_list, division_shape, stream_list):
        super(upsample_0, self).__init__()
        self.stride = s
        self.padding = p
        self.device_list = device_list
        self.division_shape = division_shape
        self.stream_list = stream_list
        self.relu = nn.ReLU(True)
        self.norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, weight, bias):
        relu_x = self.relu(x)
        convt_x = manual_convtranspose2d_multi_gpu(
            relu_x, 
            weight, 
            bias, 
            stride=self.stride, 
            padding=self.padding, 
            device_list=self.device_list,
            division_shape=self.division_shape,
            stream_list=self.stream_list
        )
        norm_x = self.norm(convt_x)
        return norm_x
    
class last_conv(nn.Module):
    def __init__(self, out_channels, s_tr, p_tr, s_c, p_c, device_list, division_shape, stream_list):
        super(last_conv, self).__init__()
        self.device_list = device_list
        self.division_shape = division_shape
        self.stream_list = stream_list
        self.stride_tr = s_tr
        self.padding_tr = p_tr
        self.stride_c = s_c
        self.padding_c = p_c
        self.tanh = nn.Tanh()
        self.norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, weight_tr, bias_tr, weight_c, bias_c):
        convt_x = manual_convtranspose2d_multi_gpu(
            x, 
            weight_tr, 
            bias_tr, 
            stride=self.stride_tr, 
            padding=self.padding_tr, 
            device_list=self.device_list,
            division_shape=self.division_shape,
            stream_list=self.stream_list
        )
        norm_x = self.norm(convt_x)
        tanh_x = self.tanh(norm_x)
        conv_x = manual_conv2d_multi_gpu(
            tanh_x, 
            weight_c, 
            bias_c, 
            stride=self.stride_c, 
            padding=self.padding_c, 
            device_list=self.device_list,
            division_shape=self.division_shape,
            stream_list=self.stream_list
        )
        return conv_x

class UNet(nn.Module):
    def __init__(self, kc, inc, ouc, device_list=[torch.device('cuda:0')], division_shape=(1,1)):
        super(UNet, self).__init__()
        stream_list = [torch.cuda.Stream(device) for device in device_list]
        self.stream_list = stream_list
        self.device_list = device_list
        self.division_shape = division_shape

        self.weight_s0 = nn.Parameter(torch.zeros(1*kc, inc, 3, 3))
        self.bias_s0 = nn.Parameter(torch.zeros(1*kc))
        
        self.s = downsample( 1*kc, 1,1, device_list, division_shape, stream_list)
        self.weight_s  = nn.Parameter(torch.zeros(1*kc, 1*kc, 3, 3))
        
        self.up = upsample( 1*kc, 2,1, device_list, division_shape, stream_list)
        self.weight_up = nn.Parameter(torch.zeros(2*kc, 1*kc, 4, 4))

        self.up0 = upsample_0( 1*kc, 1,1, device_list, division_shape, stream_list)
        self.weight_up0 = nn.Parameter(torch.zeros(2*kc, 1*kc, 3, 3))
        self.bias_up0 = nn.Parameter(torch.zeros(1*kc))
        
        self.last_Conv = last_conv(1*kc, 1,1, 1,0, device_list, division_shape, stream_list)
        self.weight_last_tr = nn.Parameter(torch.zeros(kc+inc, 1*kc, 3, 3))
        self.bias_last_tr = nn.Parameter(torch.zeros(1*kc))
        self.weight_last_c = nn.Parameter(torch.zeros(ouc, 1*kc, 1, 1))
        self.bias_last_c = nn.Parameter(torch.zeros(ouc))
        
        print("multi_gpu")
        self.init_weight()
    def init_weight(self):
        nn.init.kaiming_normal_(self.weight_s, mode='fan_out')
        nn.init.kaiming_normal_(self.weight_up, mode='fan_in')
        for w in self.modules():
            #判断层并且传参
            if isinstance(w, nn.Conv2d):
                #权重初始化
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
    def forward(self, x):
        x = x.to(torch.device('cuda:0'))
        layer_num = x.size(2).bit_length() - 1
        layer_out = [manual_conv2d_multi_gpu(x, self.weight_s0, self.bias_s0, 1, 1, self.device_list, self.division_shape, self.stream_list)]
        
        for i in range(layer_num):
            layer_out.append(self.s(layer_out[-1], self.weight_s))


        layer_out.append(layer_out[-1])

        for i in range(layer_num):
            layer_out[-2] = self.up(torch.cat([layer_out[-1], layer_out[-2]], dim=1), self.weight_up)
            layer_out.pop()

        up_0 = self.up0(
            torch.cat([layer_out[-1],layer_out[-2]], dim=1), 
            self.weight_up0,
            self.bias_up0
        )

        out  = self.last_Conv(
            torch.cat([up_0,x],dim=1), 
            self.weight_last_tr,
            self.bias_last_tr, 
            self.weight_last_c, 
            self.bias_last_c
        )
        return torch.where(out >= 0, torch.exp(out)-1, -torch.exp(-out)+1)
