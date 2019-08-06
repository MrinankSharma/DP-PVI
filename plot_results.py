import json
import numpy as np
import matplotlib.pyplot as plt

log_path = 'logs/tests/2019-08-06 19:05:01.737438'
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
exact_var = np.diag(np.diag(np.array(results['exact_inference']['var'])))

nat_mean_1 = []
nat_mean_2 = []
pres_1 = []
pres_2 = []
x = list(range(1, len(results['server']['params']) + 1))
kl = []

for w in results['server']['params']:
    nat_mean_1.append(w['w_nat_mean'][0])
    nat_mean_2.append(w['w_nat_mean'][1])
    pres_1.append(w['w_pres'][0])
    pres_2.append(w['w_pres'][1])

    nat_mean = np.array(w['w_nat_mean'])
    pres = np.array(w['w_pres'])
    mean = nat_mean / pres
    var = np.diag(1 / pres)
    kl.append(kl_mvn(mean, var, exact_mean, exact_var))
print(min(kl))

plt.figure()
plt.subplots_adjust(hspace=0.6)

plt.subplot(3, 1, 1)
plt.plot(x, nat_mean_1, label='natural mean 1')
plt.plot(x, nat_mean_2, label='natural mean 2')
plt.plot(x, pres_1, label='precision 1')
plt.plot(x, pres_2, label='precision 2')
plt.xlabel('iteration', fontsize = 12)
plt.ylabel('parameter values', fontsize = 12)
plt.legend()
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.autoscale(enable=True, axis='x', tight=True)
#plt.grid(alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(x, kl, label='KL')
plt.xlabel('iteration', fontsize = 12)
plt.ylabel('KL value', fontsize = 12)
#plt.legend(fontsize=12)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.autoscale(enable=True, axis='x', tight=True)
#plt.grid(alpha=0.3)


epsilon = []
delta = results['client_0']['MomentAccountant'][0][1]

for p in results['client_0']['MomentAccountant']:
    epsilon.append(p[0])

plt.subplot(3, 1, 3)
plt.plot(x, epsilon, label='epsilon')
plt.title('delta = ' + str(delta))
plt.xlabel('iteration', fontsize = 12)
plt.ylabel('epsilon value', fontsize = 12)
#plt.legend(fontsize=12)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.autoscale(enable=True, axis='x', tight=True)
#plt.grid(alpha=0.3)
plt.show()
