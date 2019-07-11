import numpy as np
import os


def kl_dist(p, q, use_stats_lib=False):
    """ kullback-leibler distance

    Inputs
    ------
    -. p: discrete probability distribution 'p'
    -. q: discrete probability distribution 'q'

    Output
    ------
    -. kl

    """
    if use_stats_lib:
        import scipy.stats as stats

        kl = stats.entropy(p, q)
        return kl

    # discrete probability distribution
    assert len(p) == len(q), "p and q must have same length."

    p /= np.sum(p, axis=0)
    q /= np.sum(q, axis=0)
    kl = -np.sum(p * np.log(q / p), axis=0)

    return kl


if __name__ == "__main__":
    # path
    print("Current Dir: ", os.getcwd())

    #
    # sample
    #

    # 1) sample from wikipedia
    """ https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
    D_kl( P | Q ) = 0.0852996 
    D_kl( Q | P ) = 0.097455
    """
    p = np.array([0.36, 0.48, 0.16])
    q = np.array([0.333, 0.333, 0.333])

    # KL dist
    kl_dist(p, q, use_stats_lib=False)
    kl_dist(q, p, use_stats_lib=False)

    kl_dist(p, q, use_stats_lib=True)
    kl_dist(q, p, use_stats_lib=True)


