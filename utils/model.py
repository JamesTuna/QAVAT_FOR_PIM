from utils.layer import *

################################################# model built on QAVAT Layers #################################################

class LeNet5(nn.Module):
    def __init__(self,n_classes=10,nbits_activation=None,nbits_weight=None):
        super(LeNet5, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('c1',qConv2d(1,6,kernel_size=5,stride=1,padding=2,nbits_activation=nbits_activation,nbits_weight=nbits_weight))  # 28 28 6
        self.layers.add_module('tanh1',nn.Tanh())
        self.layers.add_module('s1',nn.AvgPool2d(kernel_size=2)) # 14 14 6

        self.layers.add_module('c2',qConv2d(6,16,kernel_size=5,stride=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight)) # 10 10 16
        self.layers.add_module('tanh2',nn.Tanh())
        self.layers.add_module('s2',nn.AvgPool2d(kernel_size=2)) # 5 5 16

        self.layers.add_module('c3',qConv2d(16,120,kernel_size=5,stride=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight)) # 1 1 120
        self.layers.add_module('tanh3',nn.Tanh())

        self.layers.add_module('flatten',nn.Flatten())

        self.layers.add_module('f1',qLinear(120, 84, nbits_activation=nbits_activation,nbits_weight=nbits_weight))
        self.layers.add_module('tanh4',nn.Tanh())
        self.layers.add_module('f2',qLinear(84, n_classes, nbits_activation=nbits_activation,nbits_weight=nbits_weight))

    def forward(self,x):
        return self.layers(x)

    def clip_paras(self,min_value=-1,max_value=1):
        # deprecated, as a dummy function to force code compatibility
        # with Optimizer class (refer to utils/Optimizer)
        pass

    def generate_variation(self,noise_std):
        for m in self.modules():
            if isinstance(m,qLinear) or isinstance(m,qConv2d):
                m.generate_variation(noise_std)




class VGG11(nn.Module):
    def __init__(self,num_classes=10,nbits_activation=None,nbits_weight=None):
        super(VGG11,self).__init__()
        if nbits_weight == -1: # turned out not very helpful in A8W1, and will degrade performance a lot for A5W5
            # use proposed Batchnorm => Activation => Conv => Pool as proposed in Xnor-Net
            print('VGG: Init using BACP architecture')
            self.layers = nn.Sequential(

                        qConv2d(3,64,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                        nn.BatchNorm2d(64),
                        qConv2d(64,128,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                        nn.BatchNorm2d(128),
                        qConv2d(128,256,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(256),
                        qConv2d(256,256,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                        nn.BatchNorm2d(256),
                        qConv2d(256,512,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(512),
                        qConv2d(512,512,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                        nn.BatchNorm2d(512),
                        qConv2d(512,512,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(512),
                        qConv2d(512,512,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                        nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
                        nn.Flatten(),
                        qLinear(512, 10, nbits_activation=nbits_activation,nbits_weight=nbits_weight)
                        )
        else:
            print('VGG: Init using normal CBAP architecture')
            self.layers = nn.Sequential(

                        qConv2d(3,64,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),


                        qConv2d(64,128,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),


                        qConv2d(128,256,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        qConv2d(256,256,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),


                        qConv2d(256,512,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        qConv2d(512,512,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),


                        qConv2d(512,512,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        qConv2d(512,512,kernel_size=3,stride=1,padding=1,nbits_activation=nbits_activation,nbits_weight=nbits_weight),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                        nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
                        nn.Flatten(),
                        qLinear(512, 10, nbits_activation=nbits_activation,nbits_weight=nbits_weight)
                        )

    def forward(self,x):
        return self.layers(x)

    def clip_paras(self,min_value=-1,max_value=1):
        # deprecated, as a dummy function to force code compatibility
        # with Optimizer class (refer to utils/Optimizer)
        pass

    def generate_variation(self,noise_std):
        for m in self.modules():
            if isinstance(m,qConv2d) or isinstance(m,qLinear):
                m.generate_variation(noise_std)
