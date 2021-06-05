# PTQ of VAT models
import sys
sys.path.insert(1,'../')
from utils.model import *
from utils.loader import *
from utils.optimizer import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noise',type=float,default=0,help="std of noise applied on binarized parameters")
parser.add_argument('--testSize',type=int,default=10000,help='size of validation set')
parser.add_argument('--testSample',type=int,default=1000,help='how many noises are sampled to do validation')
parser.add_argument('--cuda',type=int,default=None,help='cuda device to train, default is None(CPU)')
parser.add_argument('--nbits_act',type=int,help='nbits of activation')
parser.add_argument('--nbits_weight',type=int,help='nbits of weight')
parser.add_argument('--tune_epoch',type=int,default=1,help='number of epochs to tune input quantization scale')
args = parser.parse_args()


device = "cpu" if args.cuda is None else "cuda:%s"%args.cuda
nbits_activation = args.nbits_act
nbits_weight = args.nbits_weight
noise = args.noise
model_name = "VGG"
VAT_ckpt = "saved/"+model_name+"_ANoneWNone_noise%.1f.ckpt"%noise
repeat = args.testSample

print("Try loading from "+VAT_ckpt+"...")
model = VGG11(nbits_weight=None,nbits_activation=None)
saved_model = torch.load(VAT_ckpt,map_location='cpu')
model.load_state_dict(saved_model)
print("load succeeded")

train_loader,test_loader = get_loader('CIFAR10',batch_size=128, test_size=10000,test_batch_size=1000)
test_loader.pin_memory = True
test_loader.num_workers = 4
train_loader.pin_memory = True
train_loader.num_workers = 8

for layer in model.modules():
    if isinstance(layer,qLinear) or isinstance(layer,qConv2d):
        layer.nbits_activation = nbits_activation
        layer.nbits_weight = nbits_weight
        layer.input_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
        layer.intervals_activation = 2**(nbits_activation-1) - 1
        layer.intervals_weight = 2**(nbits_weight-1) - 1
model.to(device)
# tune input quantization scale
model.train()
for epoch in range(args.tune_epoch):
    print("tuning input_scale: epoch %s"%epoch)
    for i, data in enumerate(train_loader, 0):
        model.train()
        x, label = data
        x = x.to(device)
        if noise > 0:
            model.generate_variation(noise_std=noise)
        output = model(x)

model.eval()
result = test(test_loader,model,noise_std=noise,
                      repeat=repeat,device=device,
                                    imgSize=32,imgFlat=False,debug=True,lossfunc=torch.nn.CrossEntropyLoss())

print("quant acc: ",result)
torch.save(result,"saved_PTQVAT/testResult_noise%.4f_ptqvat_A%sW%s_tuned%s.ckpt"%
           (noise,nbits_activation,nbits_weight,args.tune_epoch))