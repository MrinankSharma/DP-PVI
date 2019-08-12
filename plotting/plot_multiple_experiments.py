import json
import numpy as np
import matplotlib.pyplot as plt

all_log_path = [
    '../logs/tests/2019-08-11 21:00:43.228303',
    '../logs/tests/2019-08-11 21:27:26.453904',
    '../logs/tests/2019-08-11 23:50:12.086093',
    '../logs/tests/2019-08-12 00:56:37.283850',
    '../logs/tests/2019-08-12 01:23:45.349053'
]

all_results = []
for log_path in all_log_path:
    with open(log_path + '/results.json', 'r') as f:
        all_results.append(json.load(f))


def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Divergence is expressed in nats.

    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1) / np.linalg.det(S0))
    quad_term = diff.T @ iS1 @ diff

    return 0.5 * (tr_term + det_term + quad_term - N)


all_kl = []
for results in all_results:
    exact_mean = np.array(results['exact_inference']['mean'])
    exact_var = np.array(results['exact_inference']['var'])

    kl = []
    for params in results['server']['params']:
        nat_mean = np.array(params['w_nat_mean'])
        pres = np.array(params['w_pres'])
        mean = nat_mean / pres
        var = np.diag(1 / pres)
        kl.append(np.log(kl_mvn(mean, var, exact_mean, exact_var)))
    all_kl.append(kl)


all_epsilon = []
for results in all_results:
    epsilon = []
    for privacy_params in results['client_0']['MomentAccountant']:
        epsilon.append(privacy_params[0])
    all_epsilon.append(epsilon)


print(len(all_kl[0]), len(all_kl[1]), len(all_kl[2]), len(all_kl[3]), len(all_kl[4]))

plt.plot(all_epsilon[0], all_kl[0], label='C = 5')
plt.plot(all_epsilon[1], all_kl[1], label='C = 20%')
plt.plot(all_epsilon[2], all_kl[2], label='C = 40%')
plt.plot(all_epsilon[3], all_kl[3], label='C = 60%')
plt.plot(all_epsilon[4], all_kl[4], label='C = 80%')
plt.legend()
plt.xlabel('epsilon value', fontsize=12)
plt.ylabel('log of KL value', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.autoscale(enable=True, axis='x', tight=True)
plt.grid(alpha=0.3)
plt.show()
