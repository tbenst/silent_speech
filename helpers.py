import re, os, numpy as np

def sentence_to_fn(sentence, directory, ext=".wav"):
    fn = re.sub(r'[^\w\s]', '', sentence)  # remove punctuation
    fn = fn.lower().replace(' ', '_')  # lowercase with underscores
    return os.path.join(directory, fn+ext)

def load_npz_to_memory(npz_path, **kwargs):
    npz = np.load(npz_path, **kwargs)
    loaded_data = {k: npz[k] for k in npz}
    npz.close()
    return loaded_data
