
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

torch.manual_seed(0);

# Here's a simple CNN and loss function:

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output

def loss_fn(predictions, targets):
    return F.nll_loss(predictions, targets)

device = 'cuda'

num_models = 10
batch_size = 64
data = torch.randn(batch_size, 1, 28, 28, device=device)

targets = torch.randint(10, (64,), device=device)

model = SimpleCNN().to(device=device)
predictions = model(data) # move the entire mini-batch through the model

loss = loss_fn(predictions, targets)
loss.backward() # back propogate the 'average' gradient of this mini-batch

for name, param in model.named_parameters():
    print(name)
    print(param.grad.shape)

def compute_grad(sample, target):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_fn(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(data, targets):
    """ manually process each sample with per sample gradient """
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

per_sample_grads = compute_sample_grads(data, targets)

print(per_sample_grads[0].shape)
print(per_sample_grads[1].shape)

from functorch import make_functional_with_buffers, vmap, grad

fmodel, params, buffers = make_functional_with_buffers(model)

print(fmodel)

for x in params:
  print(f"{x.shape}")

print(f"\n{type(params)}")


def compute_loss_stateless_model (params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = fmodel(params, buffers, batch)
    loss = loss_fn(predictions, targets)
    return loss

ft_compute_grad = grad(compute_loss_stateless_model)

ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
print('per_sample_grads len=', len(per_sample_grads))
print('ft_per_sample_grads len=', len(ft_per_sample_grads))
print('per_sample_grads[0].shape=', per_sample_grads[0].shape)
print('ft_per_sample_grads[0]=', ft_per_sample_grads[0].shape)

# we can double check that the results using functorch grad and vmap match the results of hand processing each one individually:
# for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads):
#     assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=3e-2, rtol=1e-4)

#assertion error! why?

def get_perf(first, first_descriptor, second, second_descriptor):
  """  takes torch.benchmark objects and compares delta of second vs first. """
  second_res = second.times[0]
  first_res = first.times[0]

  gain = (first_res-second_res)/first_res
  if gain < 0: gain *=-1
  final_gain = gain*100

  print(f" Performance delta: {final_gain:.4f} percent improvement with {first_descriptor} ")

from torch.utils.benchmark import Timer

without_vmap = Timer( stmt="compute_sample_grads(data, targets)", globals=globals())
with_vmap = Timer(stmt="ft_compute_sample_grad(params, buffers, data, targets)",globals=globals())
no_vmap_timing = without_vmap.timeit(100)
with_vmap_timing = with_vmap.timeit(100)

print(f'Per-sample-grads without vmap {no_vmap_timing}')
print(f'Per-sample-grads with vmap {with_vmap_timing}')

get_perf(with_vmap_timing, "vmap", no_vmap_timing,"no vmap" )