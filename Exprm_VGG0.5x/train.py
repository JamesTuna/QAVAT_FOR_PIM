import sys
sys.path.insert(1,'../')
from utils.modelHalved import *
from utils.loader import *
from utils.optimizer import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--nbits_act', type=int, default=None, help='number of bits to quantize activations')
parser.add_argument('--nbits_weight', type=int, default=None, help='number of bits to quantize weights')
parser.add_argument('--noise', type=float, default=0, help='weight variation')
parser.add_argument('--epochs', type=int, default=200, help='how many epochs to train')
parser.add_argument('--decay_ep', type=int, default=100, help='how many epochs to decay lr')
parser.add_argument('--lr', type=float, default=1e-4, help='initial lr')
parser.add_argument('--valSize', type=int, default=10000, help='validation set size')
parser.add_argument('--valSample', type=int, default=1, help='how many test in each validation')
args = parser.parse_args()

nbits_activation = args.nbits_act
nbits_weight = args.nbits_weight
noise = args.noise
lr = args.lr
decay_ep = args.decay_ep
epochs = args.epochs
valSize = args.valSize
valSample = args.valSample


model = VGG11(nbits_activation=nbits_activation,nbits_weight=nbits_weight)

config = {"noise_std":noise, "quantization":'binary',
            "device":"cuda:%s"%0,
            "lr":lr, "decay_ep":decay_ep, "decay_ratio":0.1,
            "epochs":epochs,
            "valPerEp":10,"valSize":valSize,"valSample":valSample,
            "optimizer":"Adam"}

trial_name = 'VGG_A%sW%s_noise%s'%(nbits_activation,nbits_weight,config['noise_std'])
config['trial_name'] = trial_name
# data prep
train_loader,test_loader = get_loader('CIFAR10',batch_size=128, test_size=10000,test_batch_size=1000)
train_loader.pin_memory = True
train_loader.num_workers = 16
test_loader.pin_memory = True
test_loader.num_workers = 8

# model prep
model = model.to(config['device'])

# Train
lossfunc = nn.CrossEntropyLoss()
train(model,train_loader,test_loader,config,imgSize=32,imgFlat=False,lossfunc=lossfunc,printPerEpoch=1)

# save
torch.save(model.state_dict(),"saved/VGG_A%sW%s_noise%s.ckpt"%(nbits_activation,nbits_weight,config['noise_std']))
