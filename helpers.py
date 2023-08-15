import re, os

def sentence_to_fn(sentence, directory, ext=".wav"):
    fn = re.sub(r'[^\w\s]', '', sentence)  # remove punctuation
    fn = fn.lower().replace(' ', '_')  # lowercase with underscores
    return os.path.join(directory, fn+ext)
