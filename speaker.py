# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:04:02 2019

@author: hw
"""

import numpy as np


inter_speaker_dist = np.array([2,3,4,5,3,4,5,6,3,4])
intra_speakrt_dist = np.array([7,8,9,7,9,8,5,7,9,8,6])

# intra speaker
u = np.mean(intra_speakrt_dist)
var = np.var(intra_speakrt_dist)

#inter speaker
u_ = np.mean(inter_speaker_dist)
var_ = np.var(inter_speaker_dist)


ts = 0
if u - 3*var > u_ + 3*var_:
    ts = u - 3*var
    print("u - 3var > u_ + 3var_, th = ", ts)
else:
    ts = (u*var_ + u_*var)/(var + var_)
    print("otherwise, th = ", ts)

# false accept
fa = np.where(inter_speaker_dist > ts)

#false reject
fr = np.where(intra_speakrt_dist > ts)

far = len(fa) / len(inter_speaker_dist)
frr = len(fr) / len(intra_speakrt_dist)


print("false reject rate: ", len(fr) , '/', len(intra_speakrt_dist), '=' ,frr)
print("false accept rate: ", len(fa) , '/', len(inter_speaker_dist), '=' ,far)
