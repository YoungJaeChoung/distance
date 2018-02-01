import numpy as np
import matplotlib.pyplot as plt

def KLD(pk, qk):
	pk /= pk.sum()
	qk /= qk.sum()
	KLD = sum(pk * np.log(qk/pk))
	return(KLD)


p = np.random.normal(0, 1, 10000)
q = np.random.normal(0, 1, 10000)

plt.hist(p, bins=20)


