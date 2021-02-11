import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import reduce
import operator

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.cache = torch.zeros(1, 1, 1, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        if np.product(x.shape[1:]) != np.product(self.cache.shape[1:]):
            self.cache = x
            return self.cache

        self.cache = self.relu(x + self.cache)
        return self.cache
        

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, input):
        return input.reshape(input.shape[0], -1)

    
class ResNet(nn.Module):
    def __init__(self, dataset='CIFAR10', mode='resnet20', width=1, seed=26, variance=None):
        super(ResNet, self).__init__()

        self.ReLU = nn.ReLU()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.Flatten = Flatten()
        self.Identity = Identity()
        
        torch.manual_seed(seed=seed)
        
        if dataset in ['MNIST', 'FMNIST']:
            inp, oup = 1, 10
            
        elif dataset in ['SVHN', 'CIFAR10']:
            inp, oup = 3, 10
            
        elif dataset in ['CIFAR100']:
            inp, oup = 3, 100
        
        if mode == 'resnet20':
            self.net = nn.Sequential(           

                 nn.Conv2d(inp, int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity,

                 #1.1
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #1.2
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,


                 #1.3
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(32 * width), kernel_size=3, stride=2, padding=1, bias=True),
                 self.Identity,


                 #2.1
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #2.2
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #2.3
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(64 * width), kernel_size=3, stride=2, padding=1, bias=True),
                 self.Identity,



                 #3.1
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #3.2
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,


                 #3.3
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #Head
                 self.GAP,
                 self.Flatten,

                 nn.Linear(int(64 * width), oup))
        

        elif mode == 'resnet14':
            self.net = nn.Sequential(           

                 nn.Conv2d(inp, int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity,

                 #1.1
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,


                 #1.2
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(32 * width), kernel_size=3, stride=2, padding=1, bias=True),
                 self.Identity,


                 #2.1
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,


                 #2.3
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(64 * width), kernel_size=3, stride=2, padding=1, bias=True),
                 self.Identity,


                 #3.1
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #3.3
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #Head
                 self.GAP,
                 self.Flatten,

                 nn.Linear(int(64 * width), oup))

            
        elif mode == 'resnet9':
            self.net = nn.Sequential(           

                 nn.Conv2d(inp, int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity,

                 #1.1
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,
                
                 nn.MaxPool2d(2, 2),
                 nn.Conv2d(int(16 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity, 


                 #2.1
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,
                
                 nn.MaxPool2d(2, 2),
                 nn.Conv2d(int(32 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity,


                 #3.1
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,
                    
                 nn.MaxPool2d(2, 2),
                 
                #4.1
                 self.Identity, #remember
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,
                 nn.MaxPool2d(2, 2),
                 
                 nn.BatchNorm2d(int(64 * width)),

                 #Head
                 self.GAP,
                 self.Flatten,

                 nn.Linear(int(64 * width), oup))

            
            
        for layer in self.net.children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                if not isinstance(variance, type(None)):
                    nn.init.kaiming_normal_(layer.weight.data, 0.0, variance)
                else:
                    nn.init.kaiming_normal_(layer.weight)

                nn.init.constant_(layer.bias.data, 0.01)
        
    
        self.dictify()
        
    def flatten_parameters(self):
        total_parameters = []
        for name, par in self.net.named_parameters():
            total_parameters.append(par.detach().numpy().reshape(-1, 1))
        return np.vstack(total_parameters)

    def dictify(self):
        self.layer_dict = {}
        cum_sum = 0
        for index, (name, p) in enumerate(self.net.named_parameters()):
            block_size = np.product(p.cpu().detach().numpy().shape)
            self.layer_dict[f'{index}_'.zfill(5) + name] = [cum_sum, cum_sum + block_size]
            cum_sum += block_size
            
    def from_vector(self, theta):
        pos = 0
        for _, p in self.net.named_parameters():
            dim = reduce(operator.mul, p.size(), 1)
            p.data = torch.tensor(theta[pos:pos+dim], dtype=torch.float32).reshape(p.size())
            pos += dim

    def set_weights(self, net, theta):
        pos = 0
        for _, p in net.named_parameters():
            dim = reduce(operator.mul, p.size(), 1)
            p.data = torch.tensor(
                theta[
                    pos:pos+dim
                ], dtype=torch.float32
            ).reshape(p.size())
            pos += dim
        return net    

    
    def forward(self, x):
        return self.net(x)
    
class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()
        
    def forward(self, input):
        return input.reshape(input.shape[0], np.product(input.shape[1:]), 1, 1)

class MultipleShallowNets(nn.Module):
    def __init__(self, nets, kernel_size=3):
        super(MultipleShallowNets, self).__init__()
        
        self.kernel_size = kernel_size
        
        self.shallow_net = nets[0]
        self.wide_net =[]
        self.is_input = True
        
        self.N = len(nets)
        self.n_layers = len(list(self.shallow_net.named_parameters()))
        
        for index, block in enumerate(self.shallow_net.net):
            if isinstance(block, nn.Linear):
                block = self.concat_linear(block)
                weights_block = [basic_net.net[index] for basic_net in nets]
                block = self.concat_layers(weights_block, block)
                
            
            elif isinstance(block, nn.Conv2d):
                block = self.concat_convolution(block)
                weights_block =[basic_net.net[index] for basic_net in nets]
                block = self.concat_layers(weights_block, block)
            
            elif isinstance(block, Flatten):
                block = Vectorize()
                
            elif isinstance(block, nn.BatchNorm2d):
                block = self.concat_batchnorm_weights(block)
                weights_block =[basic_net.net[index] for basic_net in nets]
                block = self.concat_batchnorm(weights_block, block)
            
            self.is_input = False
            self.wide_net.append(block)
        self.net = nn.Sequential(*self.wide_net)
        del self.shallow_net    
        
    def concat_layers(self, layers, wide_layer):
        W = torch.cat([layer.weight for layer in layers])
        b = torch.cat([layer.bias for layer in layers])
        
        with torch.no_grad():
            if isinstance(layers[0], nn.Linear):
                wide_layer.weight = nn.Parameter(W.reshape(W.shape[0], W.shape[1], 1, 1))
            else:
                wide_layer.weight = nn.Parameter(W.reshape(W.shape[0], W.shape[1], self.kernel_size, self.kernel_size))
                
            wide_layer.bias = nn.Parameter(b)

        return wide_layer
    
    def concat_batchnorm(self, layers, wide_layer):
        weight, bias, running_mean, running_var = [], [], [], []
        
        for layer in layers:
            layer = layer.state_dict()
            
            weight.append(layer['weight'])
            bias.append(layer['bias'])
            running_mean.append(layer['running_mean'])
            running_var.append(layer['running_var'])
            
        weight, bias = torch.cat(weight), torch.cat(bias)  
        running_mean, running_var = torch.cat(running_mean), torch.cat(running_var)
        
        with torch.no_grad():
            wide_layer.weight = nn.Parameter(weight)
            wide_layer.bias = nn.Parameter(bias)
            
        wide_layer.running_mean = running_mean
        wide_layer.running_var = running_var

        return wide_layer
    
    def concat_batchnorm_weights(self, layer):
        return nn.BatchNorm2d(int(layer.num_features * self.N))
    
    def concat_linear(self, layer):
        if self.is_input:
            return nn.Conv2d(layer.in_features, int(layer.out_features * self.N), kernel_size=1)
        else:
            return nn.Conv2d(int(layer.in_features * self.N), int(layer.out_features * self.N),
                             kernel_size=1, groups=self.N)
    
    def get_conv_params(self, conv):
        parameters = {'in_channels': conv.in_channels, 'out_channels': conv.out_channels,
                      'kernel_size': conv.kernel_size[0], 'stride': conv.stride[0],
                      'padding': conv.padding[0],'dilation': conv.dilation[0],'groups': conv.groups,
                      'bias': (isinstance(conv.bias, type(None)) == False)}
        return parameters
    
    def concat_convolution(self, layer):
        parameters = self.get_conv_params(layer)
        
        if self.is_input:
            parameters['out_channels'] = parameters['out_channels'] * self.N
            return nn.Conv2d(**parameters)
        else:
            parameters['out_channels'] = parameters['out_channels'] * self.N
            parameters['in_channels'] = parameters['in_channels'] * self.N 
            parameters['groups'] = self.N
            return nn.Conv2d(**parameters)

    
    def forward(self, x):
        out = self.net(x)
        return out.reshape(out.shape[0], out.shape[1])