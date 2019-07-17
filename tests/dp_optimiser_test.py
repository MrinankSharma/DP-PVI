import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from src.privacy_accounting.optimizer import DPGaussianOptimiser
from src.privacy_accounting.analysis import PrivacyLedger

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
ledger = PrivacyLedger(x.shape[0], batch_size/x.shape[0])

dp_optimiser = DPGaussianOptimiser(
    l2_norm_clip=5,
    noise_multiplier=1,
    optimiser=optimiser,
    model=model,
    ledger=ledger,
    vector_loss=vec_loss,
    num_microbatches=None
)

for epoch in range(5):
    for batch_x, batch_y in dataloader:
        dp_optimiser.fit_batch(batch_x, batch_y)
        print(dp_optimiser.ledger)
