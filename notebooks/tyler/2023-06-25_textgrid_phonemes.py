##
from textgrids import TextGrid
import numpy as np, string

phoneme_inventory = ['aa','ae','ah','ao','aw','ax','axr','ay','b','ch','d','dh','dx','eh','el','em','en','er','ey','f','g','hh','hv','ih','iy','jh','k','l','m','n','nx','ng','ow','oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh','sil']

def read_phonemes(textgrid_fname, max_len=None):
    tg = TextGrid(textgrid_fname)
    phone_ids = np.zeros(int(tg['phones'][-1].xmax*86.133)+1, dtype=np.int64)
    phone_ids[:] = -1
    phone_ids[-1] = phoneme_inventory.index('sil') # make sure list is long enough to cover full length of original sequence
    for interval in tg['phones']:
        phone = interval.text.lower()
        if phone in ['', 'sp', 'spn']:
            phone = 'sil'
        if phone[-1] in string.digits:
            phone = phone[:-1]
        ph_id = phoneme_inventory.index(phone)
        phone_ids[int(interval.xmin*86.133):int(interval.xmax*86.133)] = ph_id
    assert (phone_ids >= 0).all(), 'missing aligned phones'

    if max_len is not None:
        phone_ids = phone_ids[:max_len]
        assert phone_ids.shape[0] == max_len
    return phone_ids

##
fn = "/scratch/librispeech/train-clean-100/5022/29411/5022-29411-0044.TextGrid"
read_phonemes(fn)
##
textgrid_fname = fn
tg = TextGrid(textgrid_fname)
##
1000 / 86.133 = 11.6

