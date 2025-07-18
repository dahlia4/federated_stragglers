import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


def func(s, OR, pRs0):
    return pRs0 / (pRs0 + OR**(1-s)*(1 - pRs0))

s_vec = np.random.binomial(1, 0.5, 1000)
d1 = np.random.binomial(1, 0.5, 1000)

OR = 4

pRs0 = [i * 0.1  for i in range(-100, 100)]

probs = [np.mean(func(s_vec, OR, expit(pRs0[i] * d1))) for i in range(len(pRs0))]

plt.plot(pRs0, probs)
plt.savefig("tmp.png")

