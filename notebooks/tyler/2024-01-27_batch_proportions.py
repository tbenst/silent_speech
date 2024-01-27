##
import numpy as np

# number of silent/parallel sEMG utterances
N_s = 1588
# number of vocal sEMG utterances
N_v = 5477
# number of sEMG utterances per example
# e = f_s*2 + f_v
# number of audio utterances per example
# a = f_s + f_v + f_l
# want:
# f_s: fraction of silent utterances
# f_v: fraction of vocal utterances
# f_l: fraction of librispeech utterances
# constraints:
# equal number of sEMG and audio utterances
# (1) a = e
# sEMG class balanced
# (2) f_v = 3.45 * f_s
# fraction sums to 1
# (3) f_s + f_v + f_l = 1
# (4) 0 < f_s < 1, 0 < f_v < 1, 0 < f_l < 1
#
##

# two-class case (no librispeech)
class2_f_s = 1588 / (5477 + 1588)
class2_f_v = 5477 / (5477 + 1588)
f_v_ratio = class2_f_v / class2_f_s
f_v_ratio  # ~3.45
##
# 2 * f_v / f_v_ratio + f_v = 1
# 2 * f_v / f_v_ratio = 1 - f_v
# 2 * f_v = (1 - f_v) * f_v_ratio
# 2 * f_v = f_v_ratio - f_v * f_v_ratio
# 2 * f_v + f_v * f_v_ratio = f_v_ratio
# f_v * (2 + f_v_ratio) = f_v_ratio
f_v = f_v_ratio / (2 + f_v_ratio)
f_l = (1 - f_v) / 2
f_s = f_l
a = f_s + f_v + f_l
e = 2 * f_s + f_v
assert np.isclose(a, e)
assert np.isclose(f_s + f_v + f_l, 1)
assert np.isclose(f_s * f_v_ratio, f_v)
print(f"{f_s=}\n{f_v=}\n{f_l=}")
# f_s = 0.18352016641627178
# f_v = 0.6329596671674564
# f_l = 0.18352016641627178
##
