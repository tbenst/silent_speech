##
%load_ext autoreload
%autoreload 2
##
import numpy as np
import sys, os
import jiwer, typer
import tiktoken, asyncio
from openai import OpenAI, AsyncOpenAI
from tqdm.auto import tqdm

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from data_utils import TextTransform
from pqdm.threads import pqdm
from helpers import calc_wer, get_neptune_run, nep_get, get_run_type
from data_utils import in_notebook
from time import sleep
from collections import OrderedDict
import logging

if in_notebook():
    # allow for nested event loops in jupyter notebooks
    import nest_asyncio
    nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)

text_transform = TextTransform(togglePhones=False)

app = typer.Typer()

TOKENIZERS = {k: tiktoken.encoding_for_model(k) for k in [
    "gpt-3.5-turbo-16k-0613", # what I used in August 2023. deprecated June 2024
    "gpt-3.5-turbo-1106", # most recent as of Jan 2024 
    "gpt-4-0125-preview" # most recent as of Jan 2024
]}
def num_tokens_from_string(string: str, model="gpt-4-0125-preview") -> int:
    """Returns the number of tokens in a text string."""
    tokenizer = TOKENIZERS[model]
    num_tokens = len(tokenizer.encode(string))
    return num_tokens

def clean_transcripts(transcripts):
    transcripts = list(map(text_transform.clean_text, transcripts))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    transcripts = transformation(transcripts)
    for i in range(len(transcripts)):
        if transcripts[i].startswith("the correct transcription is "):
            transcripts[i] = transcripts[i].replace("the correct transcription is ", "")

    return transcripts

def create_rescore_msg(predictions):
    rescore_msg = "\n".join([p for p in predictions])
    return rescore_msg

def get_labels_preds(run_id):
    run = get_neptune_run(run_id, project="neuro/Gaddy")
    output_directory = nep_get(run, "output_directory")
    hparams = nep_get(run, "training/hyperparams")
    togglePhones = hparams["togglePhones"]
    text_transform = TextTransform(togglePhones=togglePhones)
    num_beams = 5000
    path = os.path.join(output_directory, f"2024-01-28_top100_{num_beams}beams.npz")
    # path = os.path.join(output_directory, f"2024-01-27_top100_{num_beams}beams.npz")
    npz = np.load(path, allow_pickle=True)
    run_type = get_run_type(hparams)

    assert "emg_silent_val" in npz["dataset"]
    silent_idxs = np.where(npz["dataset"] == "emg_silent_val")[0]
    silent_predictions = npz["predictions"][silent_idxs]
    silent_beam_scores = npz["beam_scores"][silent_idxs]
    silent_labels = npz["sentences"][silent_idxs]
    non_zero = np.where(silent_labels != "")[0]
    silent_predictions = silent_predictions[non_zero]
    silent_beam_scores = silent_beam_scores[non_zero]
    silent_labels = silent_labels[non_zero]
    
    vocal_idxs = np.where(npz["dataset"] == "emg_vocal_val")[0]
    vocal_predictions = npz["predictions"][vocal_idxs]
    vocal_beam_scores = npz["beam_scores"][vocal_idxs]
    vocal_labels = npz["sentences"][vocal_idxs]
    non_zero = np.where(vocal_labels != "")[0]
    vocal_predictions = vocal_predictions[non_zero]
    vocal_beam_scores = vocal_beam_scores[non_zero]
    vocal_labels = vocal_labels[non_zero]
    
    audio_idxs = np.where(npz["dataset"] == "audio_val")[0]
    audio_predictions = npz["predictions"][audio_idxs]
    audio_beam_scores = npz["beam_scores"][audio_idxs]
    audio_labels = npz["sentences"][audio_idxs]
    non_zero = np.where(audio_labels != "")[0]
    audio_predictions = audio_predictions[non_zero]
    audio_beam_scores = audio_beam_scores[non_zero]
    audio_labels = audio_labels[non_zero]

    librispeech_idxs = np.where(npz["dataset"] == "librispeech_val")[0]
    librispeech_predictions = npz["predictions"][librispeech_idxs]
    librispeech_beam_scores = npz["beam_scores"][librispeech_idxs]
    librispeech_labels = npz["sentences"][librispeech_idxs]
    non_zero = np.where(librispeech_labels != "")[0]
    librispeech_beam_scores = librispeech_beam_scores[non_zero]
    librispeech_labels = librispeech_labels[non_zero]
    librispeech_predictions = librispeech_predictions[non_zero]
    return silent_predictions, silent_labels, \
        vocal_predictions, vocal_labels, \
        audio_predictions, audio_labels, \
        librispeech_predictions, librispeech_labels
        


def completion_coroutine(sys_msg, user_msg, model="gpt-3.5-turbo-16k-0613"):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        seed=20240130
    )

async def gather_completions(coroutines, n_jobs=3):
    msgs = []
    for i in tqdm(range(0, len(coroutines), n_jobs), desc="Processing completions"):
        batch = coroutines[i:i+n_jobs]
        responses = await asyncio.gather(*batch)
        msgs.extend([response.choices[0].message.content for response in responses])
    return msgs

def batch_completions(predictions, sys_msg, n_jobs=3,
                      model="gpt-3.5-turbo-16k-0613"):
    coroutines = []
    for pred in predictions:
        rescore_msg = create_rescore_msg(pred)
        cc = completion_coroutine(sys_msg, rescore_msg, model=model)
        coroutines.append(cc)
        sleep(0.05)
    # run the asynchronous gathering function
    return asyncio.run(gather_completions(coroutines, n_jobs=n_jobs))

DIRECT_SYS_MSG = """Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. Choose the transcription that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond with the chosen transcription only, without any introductory text."""
##
# crossCon + DTW 256k
run_id = "GAD-984" # 20.63% val
# run_id = "GAD-986" # 21.26% val
# run_id = "GAD-987" # 21.45% val
# run_id = "GAD-988" # 21.63% val
# run_id = "GAD-983" # 21.82% val

# # crossCon 256k
# run_id = "GAD-940" # 21.66% val
# run_id = "GAD-938" # 22.24% val
# run_id = "GAD-941" # 22.56% val
# run_id = "GAD-937" # 22.61% val
# run_id = "GAD-939" # 23.14% val
n_jobs = 8

(
    silent_pred, silent_labels,
    vocal_pred, vocal_labels,
    audio_pred, audio_labels,
    librispeech_pred, librispeech_labels
) = get_labels_preds(run_id)

silent_best_pred = np.array([p[0] for p in silent_pred])
vocal_best_pred = np.array([p[0] for p in vocal_pred])
audio_best_pred = np.array([p[0] for p in audio_pred])
librispeech_best_pred = np.array([p[0] for p in librispeech_pred])

client = AsyncOpenAI(
    max_retries=100,
    timeout=15,
)

run_silent_preds = OrderedDict()
run_silent_labels = OrderedDict()
run_vocal_preds = OrderedDict()
run_vocal_labels = OrderedDict()
run_audio_preds = OrderedDict()
run_audio_labels = OrderedDict()
run_librispeech_preds = OrderedDict()
run_librispeech_labels = OrderedDict()

for r in [
    "GAD-984",
    "GAD-986",
    "GAD-987",
    "GAD-988",
    "GAD-983",
    "GAD-940",
    "GAD-938",
    "GAD-941",
    "GAD-937",
    "GAD-939",
]:
    (
        silent_pred, silent_labels,
        vocal_pred, vocal_labels,
        audio_pred, audio_labels,
        librispeech_pred, librispeech_labels
    ) = get_labels_preds(r)
    run_silent_preds[r] = silent_pred
    run_silent_labels[r] = silent_labels
    run_vocal_preds[r] = vocal_pred
    run_vocal_labels[r] = vocal_labels
    run_audio_preds[r] = audio_pred
    run_audio_labels[r] = audio_labels
    run_librispeech_preds[r] = librispeech_pred
    run_librispeech_labels[r] = librispeech_labels

# Calculate WER for baseline
silent_wer = calc_wer(silent_best_pred, silent_labels, text_transform)
typer.echo(f"Baseline silent EMG WER: {silent_wer * 100:.2f}%")
vocal_wer = calc_wer(vocal_best_pred, vocal_labels, text_transform)
typer.echo(f"Baseline vocal EMG WER: {vocal_wer * 100:.2f}%")
audio_wer = calc_wer(audio_best_pred, audio_labels, text_transform)
typer.echo(f"Baseline audio WER: {audio_wer * 100:.2f}%")
librispeech_wer = calc_wer(librispeech_best_pred, librispeech_labels, text_transform)
typer.echo(f"Baseline librispeech WER: {librispeech_wer * 100:.2f}%")
##
#### Chain of Reasoning ####
# 22.11% with GPT-3.5, and 2 examples don't conform (10 examples)

def cor_clean_transcripts(transcripts):
    ret = []
    for transcript in transcripts:
        # split on 'TRANSCRIPT: '
        t = transcript.split("TRANSCRIPT: ")[-1]
        # remove leading and trailing whitespace
        t = t.strip()
        ret.append(t)
    ret = list(map(text_transform.clean_text, ret))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    ret = transformation(ret)
    return ret

def chain_of_reasoning_LISA(preds, labels, model):
    sys_msg = "Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. Begin your response with a Chain of Reasoning, explaining your analysis and decision-making process in choosing the most accurate transcription. After your analysis, clearly indicate your final choice with the cue 'TRANSCRIPT: '. Ensure the transcription you choose is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond first with your reasoning, followed by 'TRANSCRIPT: ' and then the chosen transcription."

    lisa_predictions = batch_completions(
        [s[:10] for s in preds],
        sys_msg, model=model, n_jobs=5)


    bad_performance = []
    for i,text in enumerate(lisa_predictions):
        if "TRANSCRIPT: " not in text:
            bad_performance.append(i)

    # we give Chain of Reasoning more than it's fair share of leeway here
    # other approaches are better so this is really just for didactic purposes
    # to show that it's not a good idea for this task
    assert len(bad_performance) < 10 # allow 5% task failure rate
    lisa_filt_predictions = [p for i,p in enumerate(lisa_predictions) if i not in bad_performance]
    filt_labels = [l for i,l in enumerate(labels) if i not in bad_performance]
    lisa_wer = calc_wer(cor_clean_transcripts(lisa_filt_predictions), filt_labels, text_transform)
    typer.echo(f"{run_id} WER with Chain of Reasoning ({model}) and excluding {bad_performance} lazy responses: {lisa_wer * 100:.2f}%")
    
    return lisa_predictions

cor_gpt3_preds = chain_of_reasoning_LISA(silent_pred, silent_labels, "gpt-3.5-turbo-16k-0613")
cor_gpt4_preds = chain_of_reasoning_LISA(silent_pred, silent_labels, "gpt-4-0125-preview")

##
def direct_LISA(preds, labels, model, N=10):
    assert len(preds) == len(labels), f"{len(preds)=} {len(labels)=}"
    lisa_predictions = batch_completions(
        [s[:N] for s in preds],
        DIRECT_SYS_MSG, model=model, n_jobs=5)
    
    try:
        lisa_wer = calc_wer(cor_clean_transcripts(lisa_predictions), labels, text_transform)
        typer.echo(f"{run_id} WER with direct {N=} ({model}): {lisa_wer * 100:.2f}%")
    except Exception as e:
        typer.echo(f"Error calculating WER: {e}")
    
    return lisa_predictions

direct10_gpt3_preds = direct_LISA(silent_pred, silent_labels, "gpt-3.5-turbo-16k-0613")
direct100_gpt3_preds = direct_LISA(silent_pred, silent_labels, "gpt-3.5-turbo-16k-0613", N=100)
direct10_gpt4_preds = direct_LISA(silent_pred, silent_labels, "gpt-4-0125-preview")
direct10_gpt4_preds = direct_LISA(silent_pred, silent_labels, "gpt-4-0125-preview", N=100)
##
#### Ensemble top1 ####
# for each entry, stack the top pred for each run
ensemble_top1_silent_preds = []
for i in range(len(silent_pred)):
    ensemble_top1_silent_preds.append(np.stack([p[i][0] for p in run_silent_preds.values()]))

ensemble_top1_vocal_preds = []
for i in range(len(vocal_pred)):
    ensemble_top1_vocal_preds.append(np.stack([p[i][0] for p in run_vocal_preds.values()]))
    
ensemble_top1_audio_preds = []
for i in range(len(audio_pred)):
    ensemble_top1_audio_preds.append(np.stack([p[i][0] for p in run_audio_preds.values()]))

ensemble_top1_librispeech_preds = []
for i in range(len(librispeech_pred)):
    ensemble_top1_librispeech_preds.append(np.stack([p[i][0] for p in run_librispeech_preds.values()])) 
##
# 5.5%
# ensemble_top1_gpt3_preds = direct_LISA(ensemble_top1_vocal_preds, silent_labels, "gpt-3.5-turbo-16k-0613")
# 3.62%
# ensemble_top1_gpt3_preds = direct_LISA(ensemble_top1_audio_preds, silent_labels, "gpt-3.5-turbo-16k-0613")
# 6.33% 
ensemble_top1_gpt3_preds = direct_LISA(ensemble_top1_librispeech_preds, librispeech_labels, "gpt-3.5-turbo-16k-0613")
# ensemble_top1_gpt3_preds = direct_LISA(ensemble_top1_silent_preds, silent_labels, "gpt-3.5-turbo-16k-0613")
# ensemble_top1_gpt4_preds = direct_LISA(ensemble_top1_silent_preds, silent_labels, "gpt-4-0125-preview")
##
# ensemble_silent_preds = []

n_per_model = 10
ensemble_top10_silent_preds = [[] for _ in range(len(silent_pred))]
# stack top 10 preds for each run (14.55% WER on GPT-3; worse than top1 of ~12%)
# for preds in run_silent_preds.values():
#     for i,topk in enumerate(preds):
#         ensemble_top10_silent_preds[i].extend(topk[:n_per_model])

for i in range(len(silent_pred)):
    for n in range(n_per_model):
        for preds in run_silent_preds.values():
            try:
                ensemble_top10_silent_preds[i].append(preds[i][n])
            except:
                print(f"no prediction for {i} {n}")
ensemble_top10_silent_preds = [np.stack(v) for v in ensemble_top10_silent_preds]

ensemble_top10_silent_preds = np.array(ensemble_top10_silent_preds)
ensemble_top10_gpt3_preds = direct_LISA(ensemble_top10_silent_preds, silent_labels, "gpt-3.5-turbo-16k-0613")
ensemble_top10_gpt4_preds = direct_LISA(ensemble_top10_silent_preds, silent_labels, "gpt-4-0125-preview")
##
#### FINETUNING dataset ####
import json

def save_finetuning_dset(preds, labels, save_path):
    dset = [(create_rescore_msg(p), l) for p,l in zip(preds, labels)]

    # Convert to JSONL format
    jsonl_data = []
    for user_msg, assistant_msg in dset:
        jsonl_data.append({
            "messages": [
                {"role": "system", "content": DIRECT_SYS_MSG},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        })

    # Save as a JSONL file
    jsonl_path = save_path
    with open(jsonl_path, 'w') as f:
        for entry in jsonl_data:
            json.dump(entry, f)
            f.write('\n')

    return jsonl_path

top1_jsonl_path = save_finetuning_dset(ensemble_top1_silent_preds[:100], silent_labels[:100],
    "../../fine_tuning_data/2024-01-30_ensemble_top1.jsonl")
top10_jsonl_path = save_finetuning_dset(ensemble_top10_silent_preds[:100], silent_labels[:100],
    "../../fine_tuning_data/2024-01-30_ensemble_top10.jsonl")
top1_vocal_path = save_finetuning_dset(ensemble_top1_vocal_preds[:100], silent_labels[:100],
    "../../fine_tuning_data/2024-01-30_ensemble_top1_vocal.jsonl")
top1_librispeech_path = save_finetuning_dset(ensemble_top1_librispeech_preds[:270], librispeech_labels[:270],
    "../../fine_tuning_data/2024-01-30_ensemble_top1_librispeech.jsonl")
##
# upload finetuning data
from openai import OpenAI
sync_client = OpenAI()

for path in [top1_jsonl_path, top10_jsonl_path, top1_vocal_path, top1_librispeech_path]:
    with open(path, "rb") as f:
        sync_client.files.create(
        file=f,
        purpose="fine-tune"
        )
##
# start finetuning job for top1
sync_client.fine_tuning.jobs.create(
    # check GUI to get file ID
    training_file="file-LPwtDDtrvDEUcbbhCj6QyNMC", 
    model="gpt-3.5-turbo-1106"
)
# vocal
sync_client.fine_tuning.jobs.create(
    # check GUI to get file ID
    training_file="file-PON6gvUaIVV4svNtZOQZR9md", 
    model="gpt-3.5-turbo-1106"
)
# librispeech
sync_client.fine_tuning.jobs.create(
    # check GUI to get file ID
    training_file="file-GBQ7cuFj7L4zSlTv0BrnLuCs", 
    model="gpt-3.5-turbo-1106"
)
# start finetuning job for top10
sync_client.fine_tuning.jobs.create(
    # check GUI to get file ID
    training_file="file-7M8a66y3VWf8OxdHLo3EPMbD", 
    model="gpt-3.5-turbo-1106"
)
##
#### finetuned ensemble top1 ####
# "ft:gpt-3.5-turbo-1106:personal::8mordMqf" was fine-tuned on ensemble top1 silent (first 100)
# "ft:gpt-3.5-turbo-1106:personal::8n2fDmAy" was fine-tuned on ensemble top1 librispeech (first 270)
# 7.3% silent EMG Validation!!!!
# lisa_predictions = direct_LISA(ensemble_top1_silent_preds[100:], silent_labels[100:],
#                                "ft:gpt-3.5-turbo-1106:personal::8mordMqf")
# 8.3% WER as model finetuned on librispeech, a bit worse than using sEMG fine-tuning GPT-3.5
lisa_predictions = direct_LISA(ensemble_top1_silent_preds[100:], silent_labels[100:],
                               "ft:gpt-3.5-turbo-1106:personal::8n2fDmAy")

# lisa_predictions = direct_LISA(ensemble_top1_vocal_preds, silent_labels,
#                                "ft:gpt-3.5-turbo-1106:personal::8mordMqf")
# lisa_predictions = direct_LISA(ensemble_top1_audio_preds[100:], silent_labels[100:],
#                                "ft:gpt-3.5-turbo-1106:personal::8mordMqf")
# lisa_predictions = direct_LISA(ensemble_top1_librispeech_preds, librispeech_labels,
#                                "ft:gpt-3.5-turbo-1106:personal::8mordMqf")
# 5.3% WER, a bit worse than using sEMG fine-tuning GPT-3.5
# lisa_predictions = direct_LISA(ensemble_top1_librispeech_preds[270:], librispeech_labels[270:],
#                                "ft:gpt-3.5-turbo-1106:personal::8n2fDmAy")
##
#### finetuned ensemble top10 ####
lisa_predictions = direct_LISA(ensemble_top10_silent_preds[100:], silent_labels[100:],
    N=100, model="ft:gpt-3.5-turbo-1106:personal::8n0xqTiW")
##
# 4% vs 5.5%
calc_wer(lisa_predictions[100:], silent_labels[100:], text_transform)
##
