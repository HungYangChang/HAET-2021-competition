from inspect import signature
import copy
from collections import namedtuple, defaultdict
import time
import numpy as np
import pandas as pd
from functools import singledispatch

import torch
from torch import nn
from collections import namedtuple 
from itertools import count

#####################
# utils
#####################

class Timer():
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t
    
localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

default_table_formats = {float: '{:{w}.4f}', str: '{:>{w}s}', 'default': '{:{w}}', 'title': '{:>{w}s}'}

def table_formatter(val, is_title=False, col_width=12, formats=None):
    formats = formats or default_table_formats
    type_ = lambda val: float if isinstance(val, (float, np.float)) else type(val)
    return (formats['title'] if is_title else formats.get(type_(val), formats['default'])).format(val, w=col_width)

def every(n, col): 
    return lambda data: data[col] % n == 0

class Table():
    def __init__(self, keys=None, report=(lambda data: True), formatter=table_formatter):
        self.keys, self.report, self.formatter = keys, report, formatter
        self.log = []
        
    def append(self, data):
        self.log.append(data)
        data = {' '.join(p): v for p,v in path_iter(data)}
        self.keys = self.keys or data.keys()
        if len(self.log) == 1:
            print(*(self.formatter(k, True) for k in self.keys))
        if self.report(data):
            print(*(self.formatter(data[k]) for k in self.keys))
            
    def df(self):
        return pd.DataFrame([{'_'.join(p): v for p,v in path_iter(row)} for row in self.log])     


#####################
## data preprocessing
#####################
def preprocess(dataset, transforms):
    dataset = copy.copy(dataset) #shallow copy
    for transform in transforms:
        dataset['data'] = transform(dataset['data'])
    return dataset

@singledispatch
def normalise(x, mean, std):
    return (x - mean) / std

@normalise.register(np.ndarray) 
def _(x, mean, std): 
    #faster inplace for numpy arrays
    x = np.array(x, np.float32)
    x -= mean
    x *= 1.0/std
    return x

unnormalise = lambda x, mean, std: x*std + mean

@singledispatch
def pad(x, border):
    raise NotImplementedError

@pad.register(np.ndarray)
def _(x, border): 
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

@singledispatch
def transpose(x, source, target):
    raise NotImplementedError

@transpose.register(np.ndarray)
def _(x, source, target):
    return x.transpose([source.index(d) for d in target]) 

#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[..., y0:y0+self.h, x0:x0+self.w]

    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]
    
    def output_shape(self, shape):
        *_, H, W = shape
        return (*_, self.h, self.w)

@singledispatch
def flip_lr(x):
    raise NotImplementedError

@flip_lr.register(np.ndarray)
def _(x): 
    return x[..., ::-1].copy()

class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return flip_lr(x) if choice else x 
        
    def options(self, shape):
        return [{'choice': b} for b in [True, False]]

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x[..., y0:y0+self.h, x0:x0+self.w] = 0.0
        return x

    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]    
    

class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        data = data.copy()
        for choices, f in zip(self.choices, self.transforms):
            data = f(data, **choices[index])
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            self.choices.append(np.random.choice(t.options(x_shape), N))
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape


#####################
## dict utils
#####################

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)  

def map_nested(func, nested_dict):
    return {k: map_nested(func, v) if isinstance(v, dict) else func(v) for k,v in nested_dict.items()}

def group_by_key(items):
    res = defaultdict(list)
    for k, v in items: 
        res[k].append(v) 
    return res

#####################
## graph building
#####################
sep = '/'

def split(path):
    i = path.rfind(sep) + 1
    return path[:i].rstrip(sep), path[i:]

def normpath(path):
    #simplified os.path.normpath
    parts = []
    for p in path.split(sep):
        if p == '..': parts.pop()
        elif p.startswith(sep): parts = [p]
        else: parts.append(p)
    return sep.join(parts)

has_inputs = lambda node: type(node) is tuple

def pipeline(net):
    return [(sep.join(path), (node if has_inputs(node) else (node, [-1]))) for (path, node) in path_iter(net)]

def build_graph(net):
    flattened = pipeline(net)
    resolve_input = lambda rel_path, path, idx: normpath(sep.join((path, '..', rel_path))) if isinstance(rel_path, str) else flattened[idx+rel_path][0]
    return {path: (node[0], [resolve_input(rel_path, path, idx) for rel_path in node[1]]) for idx, (path, node) in enumerate(flattened)}    

#####################
## training utils
#####################

@singledispatch
def cat(*xs):
    raise NotImplementedError
    
@singledispatch
def to_numpy(x):
    raise NotImplementedError

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]
 
class Const(namedtuple('Const', ['val'])):
    def __call__(self, x):
        return self.val

#####################
## network visualisation (requires pydot)
#####################
class ColorMap(dict):
    palette = ['#'+x for x in (
        'bebada,ffffb3,fb8072,8dd3c7,80b1d3,fdb462,b3de69,fccde5,bc80bd,ccebc5,ffed6f,1f78b4,33a02c,e31a1c,ff7f00,'
        '4dddf8,e66493,b07b87,4e90e3,dea05e,d0c281,f0e189,e9e8b1,e0eb71,bbd2a4,6ed641,57eb9c,3ca4d4,92d5e7,b15928'
    ).split(',')]

    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]

    def _repr_html_(self):
        css = (
        '.pill {'
            'margin:2px; border-width:1px; border-radius:9px; border-style:solid;'
            'display:inline-block; width:100px; height:15px; line-height:15px;'
        '}'
        '.pill_text {'
            'width:90%; margin:auto; font-size:9px; text-align:center; overflow:hidden;'
        '}'
        )
        s = '<div class=pill style="background-color:{}"><div class=pill_text>{}</div></div>'
        return '<style>'+css+'</style>'+''.join((s.format(color, text) for text, color in self.items()))

def make_dot_graph(nodes, edges, direction='LR', **kwargs):
    from pydot import Dot, Cluster, Node, Edge
    class Subgraphs(dict):
        def __missing__(self, path):
            parent, label = split(path)
            subgraph = Cluster(path, label=label, style='rounded, filled', fillcolor='#77777744')
            self[parent].add_subgraph(subgraph)
            return subgraph
    g = Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')
    subgraphs = Subgraphs({'': g})
    for path, attr in nodes:
        parent, label = split(path)
        subgraphs[parent].add_node(
            Node(name=path, label=label, **attr))
    for src, dst, attr in edges:
        g.add_edge(Edge(src, dst, **attr))
    return g

class DotGraph():
    def __init__(self, graph, size=15, direction='LR'):
        self.nodes = [(k, v) for k, (v,_) in graph.items()]
        self.edges = [(src, dst, {}) for dst, (_, inputs) in graph.items() for src in inputs]
        self.size, self.direction = size, direction

    def dot_graph(self, **kwargs):
        return make_dot_graph(self.nodes, self.edges, size=self.size, direction=self.direction,  **kwargs)

    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format='svg').decode('utf-8')
    try:
        import pydot
        _repr_svg_ = svg
    except ImportError:
        def __repr__(self): return 'pydot is needed for network visualisation'

walk = lambda dct, key: walk(dct, dct[key]) if key in dct else key
   
def remove_by_type(net, node_type):  
    #remove identity nodes for more compact visualisations
    graph = build_graph(net)
    remap = {k: i[0] for k,(v,i) in graph.items() if isinstance(v, node_type)}
    return {k: (v, [walk(remap, x) for x in i]) for k, (v,i) in graph.items() if not isinstance(v, node_type)}


torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

@cat.register(torch.Tensor)
def _(*xs):
    return torch.cat(xs)

@to_numpy.register(torch.Tensor)
def _(x):
    return x.detach().cpu().numpy()  

@pad.register(torch.Tensor)
def _(x, border):
    return nn.ReflectionPad2d(border)(x)

@transpose.register(torch.Tensor)
def _(x, source, target):
    return x.permute([source.index(d) for d in target]) 

def to(*args, **kwargs): 
    return lambda x: x.to(*args, **kwargs)

@flip_lr.register(torch.Tensor)
def _(x):
    return torch.flip(x, [-1])


#####################
## dataset
#####################
from functools import lru_cache as cache

@cache(None)
def cifar10(root='./data'):
    try: 
        import torchvision
        download = lambda train: torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        return {k: {'data': v.data, 'targets': v.targets} for k,v in [('train', download(train=True)), ('valid', download(train=False))]}
    except ImportError:
        from tensorflow.keras import datasets
        (train_images, train_labels), (valid_images, valid_labels) = datasets.cifar10.load_data()
        return {
            'train': {'data': train_images, 'targets': train_labels.squeeze()},
            'valid': {'data': valid_images, 'targets': valid_labels.squeeze()}
        }

def cifar100(root='./data'):
    try: 
        import torchvision
        download = lambda train: torchvision.datasets.CIFAR100(root=root, train=train, download=True)
        return {k: {'data': v.data, 'targets': v.targets} for k,v in [('train', download(train=True)), ('valid', download(train=False))]}
    except ImportError:
        from tensorflow.keras import datasets
        (train_images, train_labels), (valid_images, valid_labels) = datasets.cifar100.load_data()
        return {
            'train': {'data': train_images, 'targets': train_labels.squeeze()},
            'valid': {'data': valid_images, 'targets': valid_labels.squeeze()}
        }
             
cifar100_mean, cifar100_std = [
    [0.507, 0.487, 0.441],
    [0.267, 0.256, 0.276],
]

cifar10_mean, cifar10_std = [
    (125.31, 122.95, 113.87), # equals np.mean(cifar10()['train']['data'], axis=(0,1,2)) 
    (62.99, 62.09, 66.70), # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
]

cifar10_classes= 'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck'.split(', ')



#####################
## data loading
#####################

class DataLoader():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)

#GPU dataloading
chunks = lambda data, splits: (data[start:end] for (start, end) in zip(splits, splits[1:]))

even_splits = lambda N, num_chunks: np.cumsum([0] + [(N//num_chunks)+1]*(N % num_chunks)  + [N//num_chunks]*(num_chunks - (N % num_chunks)))

def shuffled(xs, inplace=False):
    xs = xs if inplace else copy.copy(xs) 
    np.random.shuffle(xs)
    return xs

def transformed(data, targets, transform, max_options=None, unshuffle=False):
    i = torch.randperm(len(data), device=device)
    data = data[i]
    options = shuffled(transform.options(data.shape), inplace=True)[:max_options]
    data = torch.cat([transform(x, **choice) for choice, x in zip(options, chunks(data, even_splits(len(data), len(options))))])
    return (data[torch.argsort(i)], targets) if unshuffle else (data, targets[i])

class GPUBatches():
    def __init__(self, batch_size, transforms=(), dataset=None, shuffle=True, drop_last=False, max_options=None):
        self.dataset, self.transforms, self.shuffle, self.max_options = dataset, transforms, shuffle, max_options
        N = len(dataset['data'])
        self.splits = list(range(0, N+1, batch_size))
        if not drop_last and self.splits[-1] != N:
            self.splits.append(N)
     
    def __iter__(self):
        data, targets = self.dataset['data'], self.dataset['targets']
        for transform in self.transforms:
            data, targets = transformed(data, targets, transform, max_options=self.max_options, unshuffle=not self.shuffle)
        if self.shuffle:
            i = torch.randperm(len(data), device=device)
            data, targets = data[i], targets[i]
        return ({'input': x.clone(), 'target': y} for (x, y) in zip(chunks(data, self.splits), chunks(targets, self.splits)))
    
    def __len__(self): 
        return len(self.splits) - 1

#####################
## Layers
#####################

#Network
class Network(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.graph = build_graph(net)
        for path, (val, _) in self.graph.items(): 
            setattr(self, path.replace('/', '_'), val)
    
    def nodes(self):
        return (node for node, _ in self.graph.values())
    
    def forward(self, inputs):
        outputs = dict(inputs)
        for k, (node, ins) in self.graph.items():
            #only compute nodes that are not supplied as inputs.
            if k not in outputs: 
                outputs[k] = node(*[outputs[x] for x in ins])
        return outputs
    
    def half(self):
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(node, nn.BatchNorm2d):
                node.half()
        return self

class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return x

class Add(namedtuple('Add', [])):
    def __call__(self, x, y): return x + y 
    
class AddWeighted(namedtuple('AddWeighted', ['wx', 'wy'])):
    def __call__(self, x, y): return self.wx*x + self.wy*y 

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight
    
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0, bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features*self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features*self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False): #lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
        return super().train(mode)
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return nn.functional.batch_norm(
                input.view(-1, C*self.num_splits, H, W), self.running_mean, self.running_var, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W) 
        else:
            return nn.functional.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features], 
                self.weight, self.bias, False, self.momentum, self.eps)

# Losses
class CrossEntropyLoss(namedtuple('CrossEntropyLoss', [])):
    def __call__(self, log_probs, target):
        return torch.nn.functional.nll_loss(log_probs, target, reduction='none')
    
class KLLoss(namedtuple('KLLoss', [])):        
    def __call__(self, log_probs):
        return -log_probs.mean(dim=1)

class Correct(namedtuple('Correct', [])):
    def __call__(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

class LogSoftmax(namedtuple('LogSoftmax', ['dim'])):
    def __call__(self, x):
        return torch.nn.functional.log_softmax(x, self.dim, _stacklevel=5)

x_ent_loss = Network({
  'loss':  (nn.CrossEntropyLoss(reduction='none'), ['logits', 'target']),
  'acc': (Correct(), ['logits', 'target'])
})

label_smoothing_loss = lambda alpha: Network({
        'logprobs': (LogSoftmax(dim=1), ['logits']),
        'KL':  (KLLoss(), ['logprobs']),
        'xent':  (CrossEntropyLoss(), ['logprobs', 'target']),
        'loss': (AddWeighted(wx=1-alpha, wy=alpha), ['xent', 'KL']),
        'acc': (Correct(), ['logits', 'target']),
    })

trainable_params = lambda model: {k:p for k,p in model.named_parameters() if p.requires_grad}

#####################
## Optimisers
##################### 

from functools import partial

def nesterov_update(w, dw, v, lr, weight_decay, momentum):
    dw.add_(weight_decay, w).mul_(-lr)
    v.mul_(momentum).add_(dw)
    w.add_(dw.add_(momentum, v))

norm = lambda x: torch.norm(x.reshape(x.size(0),-1).float(), dim=1)[:,None,None,None]

def LARS_update(w, dw, v, lr, weight_decay, momentum):
    nesterov_update(w, dw, v, lr*(norm(w)/(norm(dw)+1e-2)).to(w.dtype), weight_decay, momentum)

def zeros_like(weights):
    return [torch.zeros_like(w) for w in weights]

def optimiser(weights, param_schedule, update, state_init):
    weights = list(weights)
    return {'update': update, 'param_schedule': param_schedule, 'step_number': 0, 'weights': weights,  'opt_state': state_init(weights)}

def opt_step(update, param_schedule, step_number, weights, opt_state):
    step_number += 1
    param_values = {k: f(step_number) for k, f in param_schedule.items()}
    for w, v in zip(weights, opt_state):
        if w.requires_grad:
            update(w.data, w.grad.data, v, **param_values)
    return {'update': update, 'param_schedule': param_schedule, 'step_number': step_number, 'weights': weights,  'opt_state': opt_state}

LARS = partial(optimiser, update=LARS_update, state_init=zeros_like)
SGD = partial(optimiser, update=nesterov_update, state_init=zeros_like)
  
#####################
## training
#####################
from itertools import chain

def reduce(batches, state, steps):
    #state: is a dictionary
    #steps: are functions that take (batch, state)
    #and return a dictionary of updates to the state (or None)
    
    for batch in chain(batches, [None]): 
    #we send an extra batch=None at the end for steps that 
    #need to do some tidying-up (e.g. log_activations)
        for step in steps:
            updates = step(batch, state)
            if updates:
                for k,v in updates.items():
                    state[k] = v                  
    return state
  
#define keys in the state dict as constants
MODEL = 'model'
LOSS = 'loss'
VALID_MODEL = 'valid_model'
OUTPUT = 'output'
OPTS = 'optimisers'
ACT_LOG = 'activation_log'
WEIGHT_LOG = 'weight_log'

#step definitions
def forward(training_mode):
    def step(batch, state):
        if not batch: return
        model = state[MODEL] if training_mode or (VALID_MODEL not in state) else state[VALID_MODEL]
        if model.training != training_mode: #without the guard it's slow!
            model.train(training_mode)
        return {OUTPUT: state[LOSS](model(batch))}
    return step

def forward_tta(tta_transforms):
    def step(batch, state):
        if not batch: return
        model = state[MODEL] if (VALID_MODEL not in state) else state[VALID_MODEL]
        if model.training:
            model.train(False)
        logits = torch.mean(torch.stack([model({'input': transform(batch['input'].clone())})['logits'].detach() for transform in tta_transforms], dim=0), dim=0)
        return {OUTPUT: state[LOSS](dict(batch, logits=logits))}
    return step

def backward(dtype=None):
    def step(batch, state):
        state[MODEL].zero_grad()
        if not batch: return
        loss = state[OUTPUT][LOSS]
        if dtype is not None:
            loss = loss.to(dtype)
        loss.sum().backward()
    return step

def opt_steps(batch, state):
    if not batch: return
    return {OPTS: [opt_step(**opt) for opt in state[OPTS]]}

def log_activations(node_names=('loss', 'acc')):
    def step(batch, state):
        if '_tmp_logs_' not in state: 
            state['_tmp_logs_'] = []
        if batch:
            state['_tmp_logs_'].extend((k, state[OUTPUT][k].detach()) for k in node_names)
        else:
            res = {k: to_numpy(torch.cat(xs)).astype(np.float) for k, xs in group_by_key(state['_tmp_logs_']).items()}
            del state['_tmp_logs_']
            return {ACT_LOG: res}
    return step

epoch_stats = lambda state: {k: np.mean(v) for k, v in state[ACT_LOG].items()}

def update_ema(momentum, update_freq=1):
    n = iter(count())
    rho = momentum**update_freq
    def step(batch, state):
        if not batch: return
        if (next(n) % update_freq) != 0: return
        for v, ema_v in zip(state[MODEL].state_dict().values(), state[VALID_MODEL].state_dict().values()):
            if not v.dtype.is_floating_point: continue #skip things like num_batches_tracked.
            ema_v *= rho
            ema_v += (1-rho)*v
    return step

default_train_steps = (forward(training_mode=True), log_activations(('loss', 'acc')), backward(), opt_steps)
default_valid_steps = (forward(training_mode=False), log_activations(('loss', 'acc')))


def train_epoch(state, timer, train_batches, valid_batches, train_steps=default_train_steps, valid_steps=default_valid_steps, 
                on_epoch_end=(lambda state: state)):
    train_summary, train_time = epoch_stats(on_epoch_end(reduce(train_batches, state, train_steps))), timer()
    valid_summary, valid_time = epoch_stats(reduce(valid_batches, state, valid_steps)), timer(include_in_total=False) #DAWNBench rules
    return {
        'train': union({'time': train_time}, train_summary), 
        'valid': union({'time': valid_time}, valid_summary), 
        'total time': timer.total_time
    }
    
def train_epoch_new(state, timer, train_batches, train_steps=default_train_steps, on_epoch_end=(lambda state: state)):
    train_summary, train_time = epoch_stats(on_epoch_end(reduce(train_batches, state, train_steps))), timer()
    return {
        'train': union({'time': train_time}, train_summary), 
        'total time': timer.total_time
    }

def test_epoch(state, timer, valid_batches, valid_steps=default_valid_steps):
    valid_summary, valid_time = epoch_stats(reduce(valid_batches, state, valid_steps)), timer()
    return {
        'valid': union({'time': valid_time}, valid_summary), 
        'total time': timer.total_time
    }


#on_epoch_end
def log_weights(state, weights):
    state[WEIGHT_LOG] = state.get(WEIGHT_LOG, [])
    state[WEIGHT_LOG].append({k: to_numpy(v.data) for k,v in weights.items()})
    return state

def fine_tune_bn_stats(state, batches, model_key=VALID_MODEL):
    reduce(batches, {MODEL: state[model_key]}, [forward(True)])
    return state

#misc
def warmup_cudnn(model, loss, batch):
    #run forward and backward pass of the model
    #to allow benchmarking of cudnn kernels 
    reduce([batch], {MODEL: model, LOSS: loss}, [forward(True), backward()])
    torch.cuda.synchronize()

#####################
## input whitening
#####################

def cov(X):
    X = X/np.sqrt(X.size(0) - 1)
    return X.t() @ X

def patches(data, patch_size=(3, 3), dtype=torch.float32):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1, c, h, w).to(dtype)

def eigens(patches):
    n,c,h,w = patches.shape
    Σ = cov(patches.reshape(n, c*h*w))
    Λ, V = torch.symeig(Σ, eigenvectors=True)
    return Λ.flip(0), V.t().reshape(c*h*w, c, h, w).flip(0)

def whitening_filter(Λ, V, eps=1e-2):
    filt = nn.Conv2d(3, 27, kernel_size=(3,3), padding=(1,1), bias=False)
    filt.weight.data = (V/torch.sqrt(Λ+eps)[:,None,None,None])
    filt.weight.requires_grad = False 
    return filt


#Network definition
def conv_bn_default(c_in, c_out, pool=None):
    block = {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'bn': BatchNorm(c_out), 
        'relu': nn.ReLU(True)
    }
    if pool: block['pool'] = pool
    return block

def residual(c, conv_bn, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), ['in', 'res2/relu']),
    }

def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), conv_bn=conv_bn_default, prep=conv_bn_default, total_class=10):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = {
        'input': (None, []),
        'prep': prep(3, channels['prep']),
        'layer1': conv_bn(channels['prep'], channels['layer1'], pool=pool),
        'layer2': conv_bn(channels['layer1'], channels['layer2'], pool=pool),
        'layer3': conv_bn(channels['layer2'], channels['layer3'], pool=pool),
        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], total_class, bias=False),
        'logits': Mul(weight),
    }
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], conv_bn)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer])       
    return n
