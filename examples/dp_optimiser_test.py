import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from src.privacy_accounting.analysis import PrivacyLedger
from src.privacy_accounting.analysis.moment_accountant import compute_log_moments_from_ledger, get_privacy_spent
from src.privacy_accounting.optimizer import DPGaussianOptimiser

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
ledger = PrivacyLedger(x.shape[0], batch_size / x.shape[0])

dp_optimiser = DPGaussianOptimiser(
    l2_norm_clip=5,
    noise_multiplier=4,
    optimiser=optimiser,
    model=model,
    ledger=ledger,
    loss_per_example=vec_loss,
    num_microbatches=None
)

for epoch in range(100):
    for batch_x, batch_y in dataloader:
        dp_optimiser.fit_batch(batch_x, batch_y)

    print(dp_optimiser.ledger.get_formatted_ledger())
    print(len(dp_optimiser.ledger.get_formatted_ledger()))
    log_moments = compute_log_moments_from_ledger(dp_optimiser.ledger.get_formatted_ledger(), 32)
    print(get_privacy_spent(32, log_moments, None, 0.00001))
