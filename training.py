
import torch
import sys
import os
from utils import *
import os.path
import torchvision

total_class = 10

# load data 
def Data_load(root='./data'):
  # CIFAR10
  download = lambda train: torchvision.datasets.CIFAR10(root=root, train=train, download=True)
  return {k: {'data': v.data, 'targets': v.targets} for k,v in [('train', download(train=True)), ('valid', download(train=False))]}

data_sampled = Data_load('./data')

# calculate mean and std of data
data_mean = np.mean(data_sampled['train']['data'], axis=(0,1,2))
data_std = np.std(data_sampled['train']['data'], axis=(0,1,2))
print (data_mean,data_std)

batch_norm = partial(GhostBatchNorm, num_splits=4, weight_freeze=True)
relu = partial(nn.CELU, alpha=0.3)

def conv_bn(c_in, c_out, pool=None):
    block = {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'bn': batch_norm(c_out), 
        'relu': relu(),
    }
    if pool: block = {'conv': block['conv'], 'pool': pool, 'bn': block['bn'], 'relu': block['relu']}
    return block



print('Downloading datasets')
dataset = map_nested(torch.tensor, data_sampled)

## if training sample = 5000, the following setting gives best result (87%acc)
epochs, ema_epochs = 60, 10
lr_schedule = PiecewiseLinear([0, 12, epochs-ema_epochs], [0, 1.0, 1e-4])
batch_size = 128
float_size = torch.float16

# data_augmentation
train_transforms = [Crop(32, 32), FlipLR()]
loss = label_smoothing_loss(0.2)

print('Starting timer')
timer = Timer(synch=torch.cuda.synchronize)

print('Preprocessing training data')
dataset = map_nested(to(device), dataset)
T = lambda x: torch.tensor(x, dtype=float_size, device=device)

transforms = [
    to(dtype=float_size),
    partial(normalise, mean=T(data_mean), std=T(data_std)),
    partial(transpose, source='NHWC', target='NCHW'), 
]


train_set = preprocess(dataset['train'], transforms + [partial(pad, border=4)])
print(f'Finished in {timer():.2} seconds')
print(train_set['data'].shape[0], ' train imgs')


# create network
model = Network(net(weight=1/16, conv_bn=conv_bn, prep=conv_bn, total_class=total_class)).to(device).half()

train_batches = GPUBatches(batch_size=batch_size, transforms=train_transforms, dataset=train_set, shuffle=True,  drop_last=False, max_options=200)

is_bias = group_by_key(('bias' in k, v) for k, v in trainable_params(model).items())
opts = [
    SGD(is_bias[False], {'lr': (lambda step: lr_schedule(step/len(train_batches))/batch_size), 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)}),
    SGD(is_bias[True], {'lr': (lambda step: lr_schedule(step/len(train_batches))*(64/batch_size)), 'weight_decay': Const(5e-4*batch_size/64), 'momentum': Const(0.9)})
]


## training
logs_train, state = Table(), {MODEL: model, VALID_MODEL: copy.deepcopy(model), LOSS: loss, OPTS: opts}
default_train_steps = (forward(training_mode=True), log_activations(('loss', 'acc')), backward(), opt_steps)
for epoch in range(epochs):
    logs_train.append(union({'epoch': epoch+1}, train_epoch_new(state, timer, train_batches,
                                                          train_steps=(*default_train_steps, update_ema(momentum=0.99, update_freq=5))
                                                         )))
      

## save network
net_save = 'HAET_model.pt'
state = {'net': model.state_dict()}
torch.save(state, net_save)
print ("Save network!")

