import json
import numpy as np
import matplotlib.pyplot as plt

log_path = '../logs/tests/2019-08-06 02:54:31.331318'
with open(log_path + '/results.json', 'r') as f:
    results = json.load(f)


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


exact_mean = np.array(results['exact_inference']['mean'])
exact_var = np.array(results['exact_inference']['var'])
nat_mean_1 = []
nat_mean_2 = []
pres_1 = []
pres_2 = []
kl = []

for params in results['server']['params']:
    nat_mean = np.array(params['w_nat_mean'])
    pres = np.array(params['w_pres'])

    nat_mean_1.append(nat_mean[0])
    nat_mean_2.append(nat_mean[1])
    pres_1.append(pres[0])
    pres_2.append(pres[1])

    mean = nat_mean / pres
    var = np.diag(1 / pres)
    kl.append(np.log(kl_mvn(mean, var, exact_mean, exact_var)))


epsilon = []
for privacy_params in results['client_0']['MomentAccountant']:
    epsilon.append(privacy_params[0])


iter = results['client_0']['times_updated']

plt.plot(iter, nat_mean_1, label='natural mean 1')
plt.plot(iter, nat_mean_2, label='natural mean 2')
plt.plot(iter, pres_1, label='precision 1')
plt.plot(iter, pres_2, label='precision 2')
plt.legend()
plt.xlabel('number of iterations', fontsize=12)
plt.ylabel('parameter values', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.autoscale(enable=True, axis='x', tight=True)
plt.grid(alpha=0.3)
plt.show()


plt.plot(iter, kl)
plt.xlabel('number of iterations', fontsize=12)
plt.ylabel('log of KL value', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.autoscale(enable=True, axis='x', tight=True)
plt.grid(alpha=0.3)
plt.show()


plt.plot(iter, epsilon)
plt.xlabel('number of iterations', fontsize=12)
plt.ylabel('epsilon value', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.autoscale(enable=True, axis='x', tight=True)
plt.grid(alpha=0.3)
plt.show()
