import torch
import torch.nn as nn
import torch.nn.functional as F

################################################# Customized Layers for QAVAT #################################################



class Bin_act(torch.autograd.Function):
    # Scaled Sign Function + STE in Xnor-Net Paper
    # Applied to quantization with bidwidth-1 activation
    @staticmethod
    def forward(self,input):
        self.save_for_backward(input)
        # scale factor (per input channel) if features are images 
        if len(input.size()) == 4:
            n = input.size(2) * input.size(3)
            alpha = 1/n * (input.abs().sum(dim=2,keepdim=True).sum(dim=3,keepdim=True))
            signed_act = torch.mul(input.sign().detach() - input.detach() + input, alpha) # STE
        # scale factor (per layer) if features are vectors
        elif len(input.size()) == 2:
            n = input.size(1)
            alpha = 1/n * (input.abs().sum(dim=1,keepdim=True))
            signed_act = torch.mul(input.sign().detach() - input.detach() + input, alpha) # STE
        return signed_act

    @staticmethod
    def backward(self,grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(1)] = 0
        grad_input[input.lt(-1)] = 0
        return grad_input

class Bin_weight(torch.autograd.Function):
    # Scaled Sign Function + STE in Xnor-Net Paper
    # Applied to quantization with bidwidth-1 weights
    @staticmethod
    def forward(self,input):
        self.save_for_backward(input)
        # scale factor (per input channel) for conv layer weights
        if len(input.size()) == 4:
            n = input.size(0) * input.size(2) * input.size(3)*input.size(1)
            alpha = 1/n * (input.abs().sum(dim=0,keepdim=True).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True).sum(dim=1,keepdim=True))
            signed_weight = torch.mul(input.sign().detach() - input.detach() + input, alpha) # STE
        # scale factor (per weight matrix) for linear layer weights
        elif len(input.size()) == 2: 
            n = input.size(0) * input.size(1)
            alpha = 1/n * (input.abs().sum(dim=0,keepdim=True).sum(dim=1,keepdim=True))
            signed_weight = torch.mul(input.sign().detach() - input.detach() + input, alpha) # STE
        return signed_weight

    @staticmethod
    def backward(self,grad_output):
        # gradient clipping
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(1)] = 0 
        grad_input[input.lt(-1)] = 0
        return grad_input



# LINEAR LAYER
class qLinear(nn.Module):
    def __init__(self,in_features,out_features,nbits_activation=8,nbits_weight=1):
        super(qLinear,self).__init__()
        layer = nn.Linear(in_features,out_features,bias=False)
        self.weight = layer.weight
        self.weight_variation = None

        self.nbits_activation = nbits_activation
        self.nbits_weight = nbits_weight
        self.smoothness = 0.99

        if self.nbits_activation is not None and self.nbits_activation > 1:
            self.input_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
            self.intervals_activation = 2**(nbits_activation-1) - 1

        if self.nbits_weight is not None and self.nbits_weight > 1:
            self.weight_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
            self.intervals_weight = 2**(nbits_weight-1) - 1


    def forward(self,x):
        # activation quantization
        if self.nbits_activation is None:
            Q_x = x
        elif self.nbits_activation == 1:
            Q_x = Bin_act.apply(x)
        else:
            new_input_scale = torch.max(x.abs()).detach()/self.intervals_activation
            if self.input_scale == 0:
                self.input_scale += new_input_scale # initalization
            if self.training: # running statistics update for input_scale, disabled in testing
                self.input_scale.data = (self.input_scale * self.smoothness + new_input_scale * (1-self.smoothness)).data
            quant_input = (x//self.input_scale).clamp(-self.intervals_activation,self.intervals_activation)
            dequant_input = quant_input * self.input_scale
            Q_x =  dequant_input.detach() - x.detach() + x # STE

        # weight quantization
        if self.nbits_weight is None:
            Q_weight = self.weight
        elif self.nbits_weight == 1:
            Q_weight = Bin_weight.apply(self.weight)
        else:
            # fixed scale deprecated
            #new_weight_scale = torch.max(self.weight.abs()).detach()/self.intervals_weight
            #if self.weight_scale == 0:
            #    self.weight_scale += new_weight_scale # initalization
            #if self.training: # running statistics update for weight_scale, disabled in testing
            #    self.weight_scale.data = (self.weight_scale * self.smoothness + new_weight_scale * (1-self.smoothness)).data
            #quant_weight = (self.weight//self.weight_scale).clamp(-self.intervals_weight,self.intervals_weight)
            #dequant_weight = quant_weight * self.weight_scale
            #Q_weight =  dequant_weight.detach() - self.weight.detach() + self.weight # STE
            
            
            # dynamic scale in training
            if self.training or self.weight_scale==0:
                # training, update scale by line search
                nLevel = self.intervals_weight
                gs_n = 10
                init_scale = (self.weight.abs().max()/nLevel).data
                end_scale = (torch.quantile(self.weight.abs(),0.5)/nLevel).data
                gs_interval = (init_scale-end_scale)/(gs_n-1)
                scales = torch.arange(init_scale,end_scale-0.1*gs_interval,-gs_interval)
                scales = scales.unsqueeze(1).unsqueeze(2).to(self.weight.device)
                weights = self.weight.unsqueeze(0)
                Q_weights = torch.round(weights/scales).clamp_(-nLevel,nLevel)
                DQ_weights = Q_weights * scales
                L2errs = ((weights-DQ_weights)**2).sum(dim=[1,2],keepdim=False)
                index = torch.argmin(L2errs)
                self.weight_scale.data = (init_scale - index * gs_interval).data
                Q_weight = DQ_weights[index].squeeze(0).detach() - self.weight.detach() + self.weight # STE
            else:
                # evaluation, scale not updated
                quant_weight = torch.round((self.weight/self.weight_scale)).clamp(-self.intervals_weight,self.intervals_weight)
                dequant_weight = quant_weight * self.weight_scale
                Q_weight =  dequant_weight.detach() - self.weight.detach() + self.weight # STE
            

        if self.weight_variation is not None:
            Q_weight = torch.mul(Q_weight,self.weight_variation) # weight variation
        out = torch.matmul(x,Q_weight.transpose(1,0))
        return out

    def generate_variation(self,noise_std=0):
        device = self.weight.device
        if noise_std == 0:
            self.weight_variation = None
        else:
            if str(device).startswith("cuda"):
                self.weight_variation = torch.cuda.FloatTensor(self.weight.size(),device=device).normal_(1,noise_std)
            else:
                self.weight_variation = torch.FloatTensor(self.weight.size()).normal_(1,noise_std)

# CONV LAYER
class qConv2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size, stride=1,padding=0,nbits_activation=8,nbits_weight=1):

        super(qConv2d,self).__init__()
        layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                            kernel_size = kernel_size, stride = stride, padding = padding,bias=False)
        self.weight = layer.weight
        self.padding = padding
        self.stride = stride
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight_variation = None
        self.nbits_activation = nbits_activation
        self.nbits_weight = nbits_weight
        self.smoothness = 0.99

        if self.nbits_activation is not None and self.nbits_activation > 1:
            self.input_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
            self.intervals_activation = 2**(nbits_activation-1) - 1

        if self.nbits_weight is not None and self.nbits_weight > 1:
            self.weight_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
            self.intervals_weight = 2**(nbits_weight-1) - 1


    def forward(self,x):
        # activation quantization
        if self.nbits_activation is None:
            Q_x = x
        elif self.nbits_activation == 1:
            Q_x = Bin_act.apply(x)
        else:
            new_input_scale = torch.max(x.abs()).detach()/self.intervals_activation
            if self.input_scale == 0:
                self.input_scale += new_input_scale # initalization
            if self.training: # running statistics update for input_scale, disabled in testing
                self.input_scale.data = (self.input_scale * self.smoothness + new_input_scale * (1-self.smoothness)).data
            quant_input = (x//self.input_scale).clamp(-self.intervals_activation,self.intervals_activation)
            dequant_input = quant_input * self.input_scale
            Q_x =  dequant_input.detach() - x.detach() + x # STE

        # weight quantization
        if self.nbits_weight is None:
            Q_weight = self.weight
        elif self.nbits_weight == 1:
            Q_weight = Bin_weight.apply(self.weight)
        else:
            # fixed scale deprecated
            #new_weight_scale = torch.max(self.weight.abs()).detach()/self.intervals_weight
            #if self.weight_scale == 0:
            #    self.weight_scale += new_weight_scale # initalization
            #if self.training: # running statistics update for weight_scale, disabled in testing
            #    self.weight_scale.data = (self.weight_scale * self.smoothness + new_weight_scale * (1-self.smoothness)).data
            #quant_weight = (self.weight//self.weight_scale).clamp(-self.intervals_weight,self.intervals_weight)
            #dequant_weight = quant_weight * self.weight_scale
            #Q_weight =  dequant_weight.detach() - self.weight.detach() + self.weight # STE
            
            # dynamic scale in training
            if self.training or self.weight_scale==0:
                # training, update scale by line search
                nLevel = self.intervals_weight
                gs_n = 10
                init_scale = (self.weight.abs().max()/nLevel).data
                end_scale = (torch.quantile(self.weight.abs(),0.5)/nLevel).data
                gs_interval = (init_scale-end_scale)/(gs_n-1)
                scales = torch.arange(init_scale,end_scale-0.1*gs_interval,-gs_interval)
                scales = scales.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(self.weight.device)
                weights = self.weight.unsqueeze(0)
                Q_weights = torch.round(weights/scales).clamp_(-nLevel,nLevel)
                DQ_weights = Q_weights * scales
                L2errs = ((weights-DQ_weights)**2).sum(dim=[1,2,3,4],keepdim=False)
                index = torch.argmin(L2errs)
                self.weight_scale.data = (init_scale - index * gs_interval).data
                Q_weight = DQ_weights[index].squeeze(0).detach() - self.weight.detach() + self.weight # STE
            else:
                # evaluation, scale not updated
                quant_weight = torch.round((self.weight/self.weight_scale)).clamp(-self.intervals_weight,self.intervals_weight)
                dequant_weight = quant_weight * self.weight_scale
                Q_weight =  dequant_weight.detach() - self.weight.detach() + self.weight # STE

        if self.weight_variation is not None:
            Q_weight = torch.mul(Q_weight,self.weight_variation) # weight variation

        out = F.conv2d(Q_x,Q_weight,bias=None,padding=self.padding, stride=self.stride)
        return out

    def generate_variation(self,noise_std=0):
        device = self.weight.device
        if noise_std == 0:
            self.weight_variation = None
        else:
            if str(device).startswith("cuda"):
                self.weight_variation = torch.cuda.FloatTensor(self.weight.size(),device=device).normal_(1,noise_std)
            else:
                self.weight_variation = torch.FloatTensor(self.weight.size()).normal_(1,noise_std)
