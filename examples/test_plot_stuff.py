import matplotlib.pyplot as plt
import numpy as np

from src.utils.sacred_retrieval import *

a = SacredExperimentAccess(database_name='test')
exps = a.get_experiments(complete=True)[-5:]
configs = [ex['config'] for ex in exps]
configs = get_dicts_key_subset(configs, ['N_iterations',
                                         'N_samples',
                                         'dataset',
                                         'optimisation_settings',
                                         'prediction_type',
                                         'prior_pres',
                                         'privacy_settings'])
configs = get_unique_dicts(configs)

grouped_results = [dict(config=config, results=a.load_artifacts(
    a.get_experiments(complete=True, additional_filter={f'config.{k}': v for k, v in config.items()}))) for config in
                   configs]

fig = plt.figure(figsize=(8, 6))

for group in grouped_results:
    config = group['config']
    results = group['results']

    MA_epsilions = []
    PLD_epsilons = []

    for result in results:
        if len(result['artifacts']) == 0: continue
        log = result['artifacts'][0]['object']
        momenets_accountant_epsilon = [entry[0] for entry in log['client_0']['MomentAccountant']]
        pld_epsilon = [entry[0] for entry in log['client_0']['PLDAccountant']]

        MA_epsilions.append(momenets_accountant_epsilon)
        pld_epsilon.append(pld_epsilon)

    plt.plot(np.mean(MA_epsilions, axis=0), c='b', label=json.dumps(config))
    # plt.fill_between(np.mean(MA_epsilions, axis=0) + np.std(MA_epsilions, axis=0), np.mean(MA_epsilions, axis=0) - np.std(MA_epsilions, axis=0), color='b', alpha=0.4)

plt.legend()
plt.show()
