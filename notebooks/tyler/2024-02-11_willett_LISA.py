##
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from helpers import direct_LISA, batch_completions, DIRECT_SYS_MSG, in_notebook
from openai import AsyncOpenAI
from glob import glob
##
if in_notebook():
    # allow for nested event loops in jupyter notebooks
    import nest_asyncio

    nest_asyncio.apply()

##
preds = [[
    "he had money i’d be happy to be known",
    "i had money i’d be happy to be a no",
    "i had money i’d be safe to buy new",
    "i had money i’d be safe to buy new",
    "he had money i’d be safe to be a no",
    "he had money i’d be safe to buy a new one",
    "if i had money i’d be happy to be a no",
    "he had money i’d be safe to buy a known",
    "if i had money i’d be happy to be a no",
    "i had money i’d be safe to be a no",
]]

client = AsyncOpenAI(
    max_retries=100,
    timeout=15,
)
model = "gpt-3.5-turbo-16k-0613"
# lisa_predictions = batch_completions(client,
#     [s for s in preds], DIRECT_SYS_MSG, model=model, n_jobs=5
# )
# lisa_predictions

##
# TEST
def read_preds_from_dir(pred_dir):
    pred_txts = list(sorted(glob(pred_dir + "/*.txt")))
    each_file = []
    for file in pred_txts:
        with open(file, "r") as f:
            each_file.append(f.read())
    # split by newline
    each_file = [s.split("\n")[:-1] for s in each_file]
    for f in each_file:
        # 1200 competition, 600 test
        assert (len(f) == 1200) or (len(f) == 600)

    preds = [[] for _ in range(1200)]
    for f in each_file:
        for i, s in enumerate(f):
            preds[i].append(s)
    return preds

pred_txts = list(sorted(glob("/oak/stanford/projects/babelfish/magneto/willett/*.txt")))
pred_txts
each_file = []
for file in pred_txts:
    with open(file, "r") as f:
        each_file.append(f.read())
# split by newline
each_file = [s.split("\n")[:-1] for s in each_file]
for f in each_file:
    assert len(f) == 1200

preds = [[] for _ in range(1200)]
for f in each_file:
    for i, s in enumerate(f):
        preds[i].append(s)

preds[1]

##
lisa_predictions = batch_completions(
    client, preds, DIRECT_SYS_MSG, model=model, n_jobs=5
)

##
# save to file
with open(
    "/oak/stanford/projects/babelfish/magneto/willett/direct_lisa_top10ensemble_competition.txt", "w"
) as f:
    for l in lisa_predictions:
        # drop , " ? . -
        new = l.replace(",", "")
        new = new.replace('"', "")
        new = new.replace("?", "")
        new = new.replace(".", "")
        new = new.replace("-", " ")
        # to lower case
        new = new.lower()
        # convert 1975 to nineteen seventy five
        new = new.replace("1975", "nineteen seventy five")
        # remove leading and trailing whitespace
        new = new.strip()
        f.write(new + "\n")

##
with open(
    "/oak/stanford/projects/babelfish/magneto/willett/direct_lisa_top10ensemble_competition.txt", "r"
) as f:
    lines = f.readlines()
    lines = [l[:-1] for l in lines]
    unique_chars = list(set("".join(lines)))
    print(np.array(unique_chars))
    print(len(np.array(unique_chars)))

##
import numpy as np
unique_chars = list(set("".join(lisa_predictions)))
np.array(unique_chars)
##
