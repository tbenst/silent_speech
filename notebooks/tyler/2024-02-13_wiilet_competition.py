##
import os, sys, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from helpers import (
    direct_LISA,
    batch_completions,
    DIRECT_SYS_MSG,
    in_notebook,
    calc_wer,
)
from data_utils import TextTransform
from openai import AsyncOpenAI
from glob import glob
from helpers import save_finetuning_dset

if in_notebook():
    # allow for nested event loops in jupyter notebooks
    import nest_asyncio

    nest_asyncio.apply()


client = AsyncOpenAI(
    max_retries=100,
    timeout=15,
)


##
def read_preds_from_dir(pred_dir, glob_pattern="/*.txt"):
    pred_txts = list(sorted(glob(pred_dir + glob_pattern)))
    each_file = []
    for file in pred_txts:
        with open(file, "r") as f:
            each_file.append(f.read())
    # split by newline
    each_file = [s.split("\n")[:-1] for s in each_file]
    the_len = len(each_file[0])
    for f in each_file:
        # 1200 competition, 600 test
        assert (len(f) == 1200) or (len(f) == 600)

    preds = [[] for _ in range(the_len)]
    for f in each_file:
        for i, s in enumerate(f):
            preds[i].append(s)
    return preds

def cleanup_lisa_pred(lisa_pred):
    "match the cleanup in the competition script"
    # drop , " ? . -
    new = lisa_pred.replace(",", "")
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
    return new

text_transform = TextTransform()
test_preds = read_preds_from_dir(
    "/oak/stanford/projects/babelfish/magneto/willett/",
    glob_pattern="testPartition_newParams_seed*.txt",
)
# glob_pattern="testPartition_seed*.txt")
truth = []
with open("/oak/stanford/projects/babelfish/magneto/willett/testPartitionTruth.txt", "r") as f:
    lines = f.readlines()
    for l in lines:
        truth.append(l[:-1])

wers = []
for i in range(len(test_preds[0])):
    wer = calc_wer([s[i] for s in test_preds], truth, text_transform)
    print(f"Seed {i} WER: {wer*100:.2f}%")
    wers.append(wer)
mean_wer = np.array(wers).mean()
print(f"Mean WER %2=0: {mean_wer*100:.2f}%")
# sorting by best to worst model
sorted_wers = np.argsort(wers)
sorted_test_preds = []
for i in range(len(test_preds)):
    sorted_test_preds.append([])
    for j in sorted_wers:
        sorted_test_preds[i].append(test_preds[i][j])

##
# save all for fine-tuning
top1_jsonl_path = save_finetuning_dset(
    sorted_test_preds,
    truth,
    "../../fine_tuning_data/2024-02-13_willet-all.jsonl",
)
from openai import OpenAI

sync_client = OpenAI()

with open(top1_jsonl_path, "rb") as f:
    sync_client.files.create(file=f, purpose="fine-tune")
##
sync_client.fine_tuning.jobs.create(
    # check GUI to get file ID
    training_file="file-T5FyP2UHX9usEWSyUQUyioXQ",
    model="gpt-3.5-turbo-1106",
)

##
comp_preds = read_preds_from_dir(
    "/oak/stanford/projects/babelfish/magneto/willett/",
    glob_pattern="/5gramLLMCompetitionSubmission*.txt",
)

sorted_comp_preds = []
for i in range(len(comp_preds)):
    sorted_comp_preds.append([])
    for j in sorted_wers:
        sorted_comp_preds[i].append(comp_preds[i][j])
##
# Sanity check that Test WER is improved
baseline_predictions = batch_completions(
    client, sorted_test_preds, DIRECT_SYS_MSG, n_jobs=5,
    model="gpt-3.5-turbo-16k-0613",
)
baseline_test_preds = [cleanup_lisa_pred(l) for l in baseline_predictions]
finetuned_wer = calc_wer(baseline_test_preds, truth, text_transform)
print(f"Test LISA WER: {finetuned_wer*100:.2f}%")

finetuned_predictions = batch_completions(
    client, sorted_test_preds, DIRECT_SYS_MSG, n_jobs=5,
    model="ft:gpt-3.5-turbo-1106:personal::8rstXRQ6",
)
finetuned_test_preds = [cleanup_lisa_pred(l) for l in finetuned_predictions]
finetuned_wer = calc_wer(finetuned_test_preds, truth, text_transform)
print(f"Test LISA WER: {finetuned_wer*100:.2f}%")

##
# save competition predictions for fine-tuned LISA
competition_predictions = batch_completions(
    client, sorted_comp_preds, DIRECT_SYS_MSG, n_jobs=5,
    model="ft:gpt-3.5-turbo-1106:personal::8rstXRQ6",
)
competition_preds = [cleanup_lisa_pred(l) for l in competition_predictions]
with open(
    "/oak/stanford/projects/babelfish/magneto/willett/finetuned_lisa_competition.txt",
    "w",
) as f:
    for l in competition_preds:
        f.write(l + "\n")

##
for i in range(25):
    print("LISA:")
    print(competition_preds[i])
    print("ENSEMBLE:")
    for pred in sorted_comp_preds[i]:
        print(pred)
##
