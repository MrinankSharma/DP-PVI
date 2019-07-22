import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sacred import Experiment
from sacred.observers import MongoObserver

from src.privacy_accounting.analysis import PrivacyLedger
from src.privacy_accounting.analysis.moment_accountant import moment_accountant as ma
from src.privacy_accounting.analysis.online_accountant import OnlineAccountant
from src.privacy_accounting.analysis.pld_accountant import pld_accountant as pld
from src.privacy_accounting.optimizer import DPGaussianOptimizer

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

    dp_optimiser = DPGaussianOptimizer(
        l2_norm_clip=privacy['l2_norm_clip'],
        noise_multiplier=privacy['noise_multiplier'],
        optimiser=optimiser,
        model=model,
        ledger=ledger,
        loss_per_example=vec_loss,
        num_microbatches=None
    )

    MAOnline = OnlineAccountant(
        ma.compute_online_privacy_from_ledger,
        accountancy_parameters={
            'max_lambda': 32,
            'target_delta': 0.00001
        }
    )

    PLDOnline = OnlineAccountant(
        pld.compute_online_privacy_from_ledger,
        accountancy_parameters={
            'target_delta': 0.00001
        }
    )

    for epoch in range(epochs):
        print(epoch)
        for batch_x, batch_y in dataloader:
            dp_optimiser.fit_batch(batch_x, batch_y)

        t0 = time.time()
        ma_privacy = MAOnline.update_privacy(ledger.get_formatted_ledger())
        t_ma = time.time() - t0
        print(f'Moment accountant privacy: {ma_privacy} {type(ma_privacy)} : {t_ma}')

        t0 = time.time()
        pld_privacy = PLDOnline.update_privacy(ledger.get_formatted_ledger())
        print(pld_privacy)
        t_pld = time.time() - t0
        print(f'PLD accountant privacy: {pld_privacy} : {t_pld}')

        if epoch % 10 == 0:
            _run.log_scalar('moment_accountant.epsilon', ma_privacy[0])
            _run.log_scalar('moment_accountant.delta', ma_privacy[1])

            _run.log_scalar('pld_accountant.epsilon', pld_privacy[0])
            _run.log_scalar('pld_accountant.delta', pld_privacy[1])
