#!/usr/env/python
'''
An implementation of an adjusted inference algorithm in:
Inference in the Space of Topological Maps: An MCMC-based Approach
https://smartech.gatech.edu/bitstream/handle/1853/38451/Ranganathan04iros.pdf
Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

import itertools
import numpy as np
import random


def draw_sample(topo):
    '''Given a starting topology, draw a proposal topology and return the proposal ratio.
    Done by randonly choosing a merge or split move among the partitions in the topology.
    Algorithm 2.
    Input:
     - T (list of tuples) topology
    Output:
     - prop (list of tuples) update topology
     - prop_ratio (float) proposal ratio
     - move_type (int 1 or 2) integer flag for move type proposed
    '''
    T = [[ii for ii in c] for c in topo]
    if np.random.uniform(0, 1, 1) <= 0.5:
        # Perform a merge move, move_type flag == 1
        move_type = 1

        if len(T) == 1:  # if there is only one set, cannot merge
            prop = T
            prop_ratio = 1.
        else:
            # find the total number of mergeable pairs of sets, Nm
            Nm = len(list(itertools.combinations(
                T, 2)))  # simply enumerate all possible pairs

            # select P and Q, the sets to merge
            Pid = random.randrange(len(T))  # select a set at random
            P = T[Pid]
            T.pop(Pid)

            Qid = random.randrange(len(T))  # select another set at random
            Q = T[Qid]
            T.pop(Qid)

            # make the merge proposal
            merge = []
            for elem in P:
                merge.append(elem)
            for elem in Q:
                merge.append(elem)

            T.append(tuple(merge))

            # Find the total number of splits in proposal topology, Ns
            Ns = 0.
            for elem in T:
                if len(elem) > 1:
                    Ns += 1.

            # calculate the proposal ratio
            prop = T
            prop_ratio = Nm * 1. / (Ns * stirling(len(merge), 2))
    else:
        # Perform a split move, move_type flag == 2
        move_type = 2

        # find the total number of splits Ns
        Ns = 0.
        for elem in T:
            if len(elem) > 1:
                Ns += 1.

        # select R and get P and Q
        options = []
        for i, r in enumerate(T):
            if len(r) <= 1:
                pass
            else:
                options.append(i)

        if Ns < 1:
            # there are only singleton sets, propose the same topo
            prop = T
            prop_ratio = 1.
        else:
            # there are at least options
            Rid = options[random.randrange(
                0,
                len(options))]  # choose a random index from the valid options
            R = T[Rid]
            T.pop(Rid)
            splitid = random.randrange(len(R))
            P = R[:splitid]
            Q = R[splitid:]
            if len(P) > 0:
                T.append(tuple(P))
            if len(Q) > 0:
                T.append(tuple(Q))

            # find the total number of sets in proposal topology, Nm
            if len(T) > 1:
                Nm = len(list(itertools.combinations(T, 2)))
                prop_ratio = 1. / Nm * Ns * stirling(len(R), 2)
            else:
                Nm = 1.
                prop_ratio = 0.0  # 1./Nm * Ns * stirling(len(R), 2)
            prop = T

    return prop, prop_ratio, move_type


def stirling(n, m):
    '''Recursive function that returns the Stirling number of the first
    kind.'''
    row = [1] + [0 for _ in range(m)]
    for i in range(1, n + 1):
        new = [0]
        for j in range(1, m + 1):
            sling = (i - 1) * row[j] + row[j - 1]
            new.append(sling)
        row = new
    return row[m]
