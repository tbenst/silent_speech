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
# file = "/oak/stanford/projects/babelfish/magneto/willett/ptOutputs_test_nbest_seed0.txt"
# with open(file, "r") as f:
#     topk = f.read().split("\n\n")
# preds = [k.split("\n") for k in topk][:-1]
# preds = [[cleanup_lisa_pred(l) for l in p] for p in preds]

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
    # find and remove all text after newline
    new = new.split("\n")[0]
    # remove leading and trailing whitespace
    new = new.strip()
    return new

def read_topk_from_nbest(file):
    with open(file, "r") as f:
        topk = f.read().split("\n\n")
    preds = [k.split("\n") for k in topk][:-1]
    preds = [[cleanup_lisa_pred(l) for l in p] for p in preds]
    assert (len(preds) == 1200) or (len(preds) == 600), f"len(preds)={len(preds)}"
    return preds

def read_topk_from_dir(pred_dir, glob_pattern="/*.txt", topK=1):
    pred_txts = list(sorted(glob(pred_dir + glob_pattern)))
    print(pred_txts)
    model_preds = []
    for file in pred_txts:
        model_preds.append(read_topk_from_nbest(file))
    N = len(model_preds[0])
    preds_to_return = [[] for _ in range(N)]
    for i, models_preds in enumerate(zip(*model_preds)):
        # models_preds is a list of predictions for each model
        longest = max(len(ps) for ps in models_preds)
        longest = min(longest, topK)
        # append predictions to list in model order of:
        # model 1, model 2, ..., model N, model 1, model 2, ..., model N, ...
        for j in range(longest):
            for model_preds in models_preds:
                if j < len(model_preds):
                    preds_to_return[i].append(model_preds[j])
    return preds_to_return

text_transform = TextTransform()
# test_preds = read_topk_from_nbest(
#     "/oak/stanford/projects/babelfish/magneto/willett/ptOutputs_test_nbest_seed0.txt"
# )

topK = 1
test_preds = read_topk_from_dir(
    "/oak/stanford/projects/babelfish/magneto/willett/nbest/",
    "ptOutputs_test_nbest_seed*.txt",
    topK=topK)
# glob_pattern="testPartition_seed*.txt")
truth = []
with open(
    "/oak/stanford/projects/babelfish/magneto/willett/testPartitionTruth.txt", "r"
) as f:
    lines = f.readlines()
    for l in lines:
        truth.append(l[:-1])

########## OPTION 1 (dev) ##########
# # we will test sorting on this (Validation set)
# wers = []
# nModels = 10
# for i in range(nModels):
#     # wer = calc_wer([s[i] for s in test_preds], truth, text_transform)
#     wer = calc_wer([s[i] for s in test_preds[1::2]], truth[1::2], text_transform)
#     print(f"Seed {i} WER: {wer*100:.2f}%")
#     wers.append(wer)
# mean_wer = np.array(wers).mean()
# assert max([len(p) for p in test_preds]) == topK * nModels
# print(f"Mean WER %2=1 {mean_wer*100:.2f}%")

# # we will sort/train on this (Finetuning set)
# wers = []
# nModels = 10
# for i in range(nModels):
#     # wer = calc_wer([s[i] for s in test_preds], truth, text_transform)
#     wer = calc_wer([s[i] for s in test_preds[::2]], truth[::2], text_transform)
#     print(f"Seed {i} WER: {wer*100:.2f}%")
#     wers.append(wer)
# mean_wer = np.array(wers).mean()
# assert max([len(p) for p in test_preds]) == topK * nModels
# print(f"Mean WER %2=0 {mean_wer*100:.2f}%")
# print("WARNING: for final run make sure to sort WER on all test data")

########## OPTION 2 (comp time) ##########
wers = []
nModels = 10
for i in range(nModels):
    # wer = calc_wer([s[i] for s in test_preds], truth, text_transform)
    wer = calc_wer([s[i] for s in test_preds], truth, text_transform)
    print(f"Seed {i} WER: {wer*100:.2f}%")
    wers.append(wer)
mean_wer = np.array(wers).mean()
assert max([len(p) for p in test_preds]) == topK * nModels
print(f"Mean WER {mean_wer*100:.2f}%")

##
# sorting by best to worst model
sorted_wers = np.argsort(wers)
sorted_test_preds = []
for i in range(len(test_preds)):
    sorted_test_preds.append([])
    for k in range(topK):
        for m in sorted_wers:
            try:
                sorted_test_preds[i].append(test_preds[i][k * nModels + m])
            except:
                pass # some models have fewer than topK predictions
        # sorted_test_preds[i].append(test_preds[i][j])

assert np.all([len(p) for p in sorted_test_preds] == \
    [len(p) for p in test_preds])

assert (np.all(sorted_test_preds[1][10:20] == \
        np.array(test_preds[1][10:20])[sorted_wers].tolist()
))
##
baseline_predictions = batch_completions(
    client,
    sorted_test_preds,
    DIRECT_SYS_MSG,
    n_jobs=5,
    model="gpt-3.5-turbo-16k-0613",
)
baseline_test_preds = [cleanup_lisa_pred(l) for l in baseline_predictions]
finetuned_wer = calc_wer(baseline_test_preds[1::2], truth[1::2], text_transform)
print(f"Test %2=1 {topK=} LISA WER: {finetuned_wer*100:.2f}%")
# top1: 13.79%
# top5: 13.04%
# top10: 12.72%
# top100: 12.88%
##
# save all for fine-tuning
top1_jsonl_path = save_finetuning_dset(
    sorted_test_preds,
    truth,
    f"../../fine_tuning_data/2024-02-15_willet-pytorch-all_top{topK}.jsonl",
)
from openai import OpenAI

sync_client = OpenAI()

with open(top1_jsonl_path, "rb") as f:
    sync_client.files.create(file=f, purpose="fine-tune")
##
sync_client.fine_tuning.jobs.create(
    # check GUI to get file ID
    # training_file="file-WhDxzMDtxrvBSo3NbA3WCFcr",
    training_file="file-FcDJGvcYgLR2Wz6CC4FwSZi9",
    model="gpt-3.5-turbo-1106",
)

##
comp_preds = read_topk_from_dir(
    "/oak/stanford/projects/babelfish/magneto/willett/nbest/",
    "ptOutputs_competition_nbest_seed*.txt",
    topK=topK,
)

sorted_comp_preds = []
for i in range(len(comp_preds)):
    sorted_comp_preds.append([])
    for j in sorted_wers:
        sorted_comp_preds[i].append(comp_preds[i][j])
##
assert topK == 1
model = "ft:gpt-3.5-turbo-1106:personal::8sniQV12"

# assert topK == 10
# model = "ftjob-ehtuoPJiNjykGf8GP3hBoH84"

# Sanity check that Test WER is improved
baseline_predictions = batch_completions(
    client,
    sorted_test_preds,
    DIRECT_SYS_MSG,
    n_jobs=5,
    model="gpt-3.5-turbo-16k-0613",
)
baseline_test_preds = [cleanup_lisa_pred(l) for l in baseline_predictions]
finetuned_wer = calc_wer(baseline_test_preds, truth, text_transform)
print(f"Test {topK=} LISA WER: {finetuned_wer*100:.2f}%")

finetuned_predictions = batch_completions(
    client,
    sorted_test_preds,
    DIRECT_SYS_MSG,
    n_jobs=5,
    # model="ft:gpt-3.5-turbo-1106:personal::8rstXRQ6",
    # model="ft:gpt-3.5-turbo-1106:personal::8sjJbyhn",
    model=model,
)
finetuned_test_preds = [cleanup_lisa_pred(l) for l in finetuned_predictions]
finetuned_wer = calc_wer(finetuned_test_preds, truth, text_transform)
print(f"Test LISA WER: {finetuned_wer*100:.2f}%")

##
# save competition predictions for fine-tuned LISA
competition_predictions = batch_completions(
    client,
    sorted_comp_preds,
    DIRECT_SYS_MSG,
    n_jobs=5,
    # model="ft:gpt-3.5-turbo-1106:personal::8rstXRQ6",
    # model="ft:gpt-3.5-turbo-1106:personal::8sjJbyhn",
    model=model
)
competition_preds = [cleanup_lisa_pred(l) for l in competition_predictions]
with open(
    f"/oak/stanford/projects/babelfish/magneto/willett/2024-02-16_top{topK}_competition.txt",
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
