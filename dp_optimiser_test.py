import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from src.privacy_accounting.optimizer import DPOptimiser
from src.privacy_accounting.dp_query import GaussianDPQuery

model = nn.Sequential(nn.Linear(1, 1))

x = np.atleast_2d(np.random.uniform(-5, 5, 1000)).T
y = x * 5 + np.atleast_2d(np.random.normal(0, 0.5))

def vec_loss(y_, y):
    return (y - y_) ** 2

dataset = data.TensorDataset(
    torch.Tensor(x),
    torch.Tensor(y)
)
batch_size = 10
dataloader = data.DataLoader(dataset, batch_size=batch_size)

optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

query = GaussianDPQuery(5., 1.)

dp_optimiser = DPOptimiser(
    optimiser=optimiser,
    model=model,
    vector_loss=vec_loss,
    dp_sum_query=query,
    microbatch_size=1
)

for epoch in range(5):
    for p in model.parameters():
        print(p)
    for batch_x, batch_y in dataloader:
        dp_optimiser.fit_batch(batch_x, batch_y)
