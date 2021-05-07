import sys,os
sys.path.insert(1,'../')
from utils.model import *
from utils.loader import *
from utils.optimizer import *
import torch.nn as nn # used to define loss function arguments passed to train()



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--noise',type=float,default=0,help="std of noise applied on binarized parameters")
parser.add_argument('--testSize',type=int,default=1000,help='size of validation set')
parser.add_argument('--testSample',type=int,default=1000,help='how many noises are sampled to do validation')
parser.add_argument('--cuda',type=int,default=None,help='cuda device to train, default is None(CPU)')
parser.add_argument('--load',type=str,help='load pretrained model if a path is given')
parser.add_argument('--nbits_act',type=int,help='nbits of activation')
parser.add_argument('--nbits_weight',type=int,help='nbits of weight')
args = parser.parse_args()

model = VGG11(nbits_weight=args.nbits_weight,nbits_activation=args.nbits_act)


_,test_loader = get_loader('CIFAR10',batch_size=100, test_size=args.testSize,test_batch_size=1000)
test_loader.pin_memory = True
test_loader.num_workers = 4

print("Try loading %s"%args.load)
saved_model = torch.load(args.load,map_location='cpu')
model.load_state_dict(saved_model)
print("load succeeded")
print("model basename: "+os.path.basename(args.load))

device = "cpu" if args.cuda is None else "cuda:%s"%args.cuda
model = model.to(device)

print("Test under noise %.4f"%args.noise)
model.eval()
result = test(test_loader,model,noise_std=args.noise,
                      repeat=args.testSample,device=device,
                                    imgSize=32,imgFlat=False,debug=True,lossfunc=torch.nn.CrossEntropyLoss())

print("test finished,",result)
torch.save(result,"saved/testResult_noise%.4f_"%(args.noise)+os.path.basename(args.load))
