##
%load_ext autoreload
%autoreload 2
##
import os, sys, json
from openai import OpenAI, AsyncOpenAI
from glob import glob
from tqdm import tqdm
import numpy as np, asyncio
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from read_emg import TESTSET_FILE
from data_utils import TextTransform
from helpers import calc_wer, get_neptune_run, nep_get, get_run_type, \
    create_rescore_msg, completion_coroutine, gather_completions, \
    batch_completions, cor_clean_transcripts, direct_LISA, \
    save_finetuning_dset, get_transcript, gather_transcripts, batch_transcripts

from data_utils import in_notebook

if in_notebook():
    # allow for nested event loops in jupyter notebooks
    import nest_asyncio

    nest_asyncio.apply()


with open(TESTSET_FILE) as f:
    testset_json = json.load(f)
    valset = testset_json["dev"]
    testset = testset_json["test"]
text_transform = TextTransform(togglePhones=False)
client = OpenAI()
async_client = AsyncOpenAI(
    max_retries=100,
    timeout=15,
)
##
voiced_parallel_data_dir = (
    "/scratch/users/tbenst/GaddyPaper/emg_data/voiced_parallel_data/"
)
json_files = glob(os.path.join(voiced_parallel_data_dir, "**", "*info.json"))
len(json_files)
##
val_wav_files = []
val_text = []
test_wav_files = []
test_text = []

for jf in json_files:
    with open(jf) as f:
        j = json.load(f)
        book = j["book"]
        sentence_index = j["sentence_index"]
        text = j["text"]
        audio_file = jf[: -len("info.json")] + "audio.wav"
        if [book, sentence_index] in valset:
            val_wav_files.append(audio_file)
            val_text.append(text)
        elif [book, sentence_index] in testset:
            test_wav_files.append(audio_file)
            test_text.append(text)
assert len(val_wav_files) == 200

assert val_text[165] == "."
val_wav_files = val_wav_files[:165] + val_wav_files[166:]
val_text = val_text[:165] + val_text[166:]


##
best_pred_text = batch_transcripts(async_client, val_wav_files, [0], n_jobs=3)

##
# one has decent predictions but diversity still really low
temps = [0.0, 0.2, 0.4, 0.6, 0.8] + [1.0] * 5
# pred_text = batch_transcripts(async_client, val_wav_files, temps, n_jobs=3)
pred_text = batch_transcripts(async_client, val_wav_files, temps, n_jobs=3)

for (
    target,
    preds,
) in zip(val_text, pred_text):
    print("-" * 80)
    print(f"target: {text_transform.clean_text(target)}")
    for t, pred in zip(temps, preds):
        p = text_transform.clean_text(pred)
        print(f"{t:.4f}: {p}")


##
best_wer = calc_wer(best_pred_text, val_text, text_transform)
print(f"Whisper Validation WER: {best_wer * 100:.2f}%")  # 2.62%

best_wer = calc_wer(best_pred_text[100:], val_text[100:], text_transform)
print(f"Whisper Validation (second 100) WER: {best_wer * 100:.2f}%")  # 2.26%


##
gpt_model = "gpt-3.5-turbo-16k-0613" # 2.54% ever so slightly better :D 
whisper_lisa_preds = direct_LISA(async_client, pred_text, val_text,
    gpt_model, text_transform)
##
# gpt_model = "gpt-3.5-turbo-16k-0613" # 2.37% slightly worse than no LISA
gpt_model = "ft:gpt-3.5-turbo-1106:personal::8o7GZcng" # 2.20% so ever so slightly better
whisper_lisa_preds = direct_LISA(async_client, pred_text[100:], val_text[100:],
    gpt_model, text_transform)

##
lisa_wer = calc_wer(
    cor_clean_transcripts(whisper_lisa_preds, text_transform), val_text, text_transform
)
print(f"WER: {lisa_wer * 100:.2f}%")

##
path = save_finetuning_dset(whisper_lisa_preds[:100], val_text[:100],
    "../../fine_tuning_data/2024-02-03_whisper_lisa_top10_gaddy_val_audio_first100.jsonl")

with open(path, "rb") as f:
    client.files.create(
    file=f,
    purpose="fine-tune"
    )
##
client.fine_tuning.jobs.create(
    # check GUI to get file ID
    training_file="file-Bf237WUYkDoI6qtH7YjMFmAF", 
    model="gpt-3.5-turbo-1106"
)
##
lisa_wer = calc_wer(
    cor_clean_transcripts(whisper_lisa_preds[100:], text_transform), val_text[100:], text_transform
)
print(f"WER: {lisa_wer * 100:.2f}%")
##
