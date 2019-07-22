import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sacred import Experiment
from sacred.observers import MongoObserver

from src.privacy_accounting.analysis import PrivacyLedger
from src.privacy_accounting.analysis.moment_accountant import compute_log_moments_from_ledger, get_privacy_spent
from src.privacy_accounting.analysis.pld_accountant import pld_accountant as pld
from src.privacy_accounting.optimizer import DPGaussianOptimiser

ex = Experiment('Test Experiment')

ex.observers.append(MongoObserver.create(
    url='localhost:9001',
    db_name='sacred'
))


@ex.config
def cfg():
    N = 1000
    batch_size = 10
    learning_rate = 0.001
    epochs = 100
    privacy = {
        "l2_norm_clip": 5,
        "noise_multiplier": 4,
        "max_delta": 0.00001,
        "max_lambda": 32
    }


ex.add_config('test_config.yaml')


@ex.automain
def main(N, batch_size, learning_rate, epochs, privacy, _run):
    model = nn.Sequential(nn.Linear(1, 1))

    x = np.atleast_2d(np.random.uniform(-5, 5, N)).T
    y = x * 5 + np.atleast_2d(np.random.normal(0, 0.5))

    def vec_loss(y_, y):
        return (y - y_) ** 2

    dataset = data.TensorDataset(
        torch.Tensor(x),
        torch.Tensor(y)
    )

    dataloader = data.DataLoader(dataset, batch_size=batch_size)

    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    ledger = PrivacyLedger(x.shape[0], batch_size / x.shape[0])

    dp_optimiser = DPGaussianOptimiser(
        l2_norm_clip=privacy['l2_norm_clip'],
        noise_multiplier=privacy['noise_multiplier'],
        optimiser=optimiser,
        model=model,
        ledger=ledger,
        loss_per_example=vec_loss,
        num_microbatches=None
    )

    for epoch in range(epochs):
        print(epoch)
        for batch_x, batch_y in dataloader:
            dp_optimiser.fit_batch(batch_x, batch_y)

        t0 = time.time()
        log_moments = compute_log_moments_from_ledger(dp_optimiser.ledger.get_formatted_ledger(), 32)
        ma_privacy = get_privacy_spent(privacy['max_lambda'], log_moments, None, privacy['max_delta'])
        t_ma = time.time() - t0
        print(f'Moment accountant privacy: {ma_privacy} : {t_ma}')

        t0 = time.time()
        pld_privacy = pld.compute_prvacy_loss_from_ledger(ledger.get_formatted_ledger(),
                                                          target_delta=privacy['max_delta'])
        t_pld = time.time() - t0
        print(f'PLD accountant privacy: {pld_privacy} : {t_pld}')

        _run.log_scalar('moment_accountant.epsilon', ma_privacy[0])
        _run.log_scalar('moment_accountant.delta', ma_privacy[1])

        _run.log_scalar('pld_accountant.epsilon', pld_privacy[0])
        _run.log_scalar('pld_accountant.delta', pld_privacy[1])
