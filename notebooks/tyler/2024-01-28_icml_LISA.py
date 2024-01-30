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
import logging

if in_notebook():
    # allow for nested event loops in jupyter notebooks
    import nest_asyncio
    nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)

text_transform = TextTransform(togglePhones=False)

app = typer.Typer()


def create_rescore_msg(predictions, scores):
    rescore_msg = "\n".join([f"{s:.3f}\t{p}" for p, s in zip(predictions, scores)])
    return rescore_msg    

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


@app.command()
def main(
    npz_file: str = typer.Argument(
        ..., help="Path to the .npz file containing the predictions and scores"
    ),
    sys_msg: str = typer.Option(None, help="System message to use as input (optional)"),
    n_jobs: int = typer.Option(3, help="Number of jobs for parallel processing"),
    baseline: bool = typer.Option(
        False, help="Calculate only the baseline WER"
    ),  # Added this option
):
    # Load the .npz file
    npz = np.load(npz_file, allow_pickle=True)

    # If sys_msg is not provided, use the default message
    if sys_msg is None:
        sys_msg = """
        Your task is automatic speech recognition. \
        Below are the candidate transcriptions along with their \
        negative log-likelihood from a CTC beam search. \
        Respond with the correct transcription, \
        without any introductory text.
        """.strip()

    # Calculate WER for baseline
    baseline_wer = calc_wer([n[0] for n in npz["predictions"]], npz["sentences"])
    typer.echo(f"Baseline WER: {baseline_wer * 100:.2f}%")

    if not baseline:
        # Get transcripts using the provided function
        transcripts = batch_predict_from_topk(
            npz["predictions"], npz["beam_scores"], sys_msg=sys_msg, n_jobs=n_jobs
        )

        # Clean and calculate WER for the transcripts
        wer = calc_wer(clean_transcripts(transcripts), npz["sentences"])
        typer.echo(
            f"Baseline WER: {baseline_wer * 100:.2f}%"
        )  # repeat due to noisy output
        typer.echo(f"Final WER: {wer * 100:.2f}%")

    # Run the application
    # if __name__ == "__main__":
    #     app()


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
print(run_type)
# If sys_msg is not provided, use the default message
sys_msg = """
Your task is automatic speech recognition. \
Below are the candidate transcriptions along with their \
negative log-likelihood from a CTC beam search. \
Respond with the correct transcription, \
without any introductory text.
""".strip()

assert "emg_silent_val" in npz["dataset"]
# silent_idxs = np.where(npz["dataset"] == "silent_emg_val")[0]
silent_idxs = np.where(npz["dataset"] == "emg_silent_val")[0]
silent_predictions = npz["predictions"][silent_idxs]
silent_beam_scores = npz["beam_scores"][silent_idxs]
silent_labels = npz["sentences"][silent_idxs]
non_zero = np.where(silent_labels != "")[0]
silent_best_predictions = np.array([p[0] for p in silent_predictions[non_zero]])
silent_beam_scores = silent_beam_scores[non_zero]
silent_labels = silent_labels[non_zero]

librispeech_idxs = np.where(npz["dataset"] == "librispeech_val")[0]
librispeech_predictions = npz["predictions"][librispeech_idxs]
librispeech_beam_scores = npz["beam_scores"][librispeech_idxs]
librispeech_labels = npz["sentences"][librispeech_idxs]
non_zero = np.where(librispeech_labels != "")[0]
librispeech_best_predictions = np.array([p[0] for p in librispeech_predictions[non_zero]])
librispeech_beam_scores = librispeech_beam_scores[non_zero]
librispeech_labels = librispeech_labels[non_zero]

# Calculate WER for baseline
baseline_wer = calc_wer(silent_best_predictions, silent_labels, text_transform)
typer.echo(f"Baseline WER: {baseline_wer * 100:.2f}%")
##
# try new API
# gpt-3.5 has topen limit of 160k/min
# gpt-4-turbo has rate limit of 300k/min
predictions = silent_predictions[0]
scores = silent_beam_scores[0]
client = AsyncOpenAI(
    max_retries=100,
    timeout=15,
)

rescore_msg = create_rescore_msg(predictions, scores)
num_tokens = num_tokens_from_string(rescore_msg) + num_tokens_from_string(sys_msg)
model="gpt-3.5-turbo-16k-0613"
# model="gpt-4-0125-preview",

num_pred = len(predictions)
if num_pred < 10:
    print(f"WARNING: only {num_pred} predictions from beam search")

print(rescore_msg)
print(num_tokens)
##
def completion_coroutine(sys_msg, user_msg, model="gpt-3.5-turbo-16k-0613"):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

async def gather_completions(coroutines, n_jobs=3):
    msgs = []
    for i in tqdm(range(0, len(coroutines), n_jobs), desc="Processing completions"):
        batch = coroutines[i:i+n_jobs]
        responses = await asyncio.gather(*batch)
        msgs.extend([response.choices[0].message.content for response in responses])
    return msgs

def batch_completions(predictions, scores, sys_msg, n_jobs=3,
                      model="gpt-3.5-turbo-16k-0613"):
    coroutines = []
    for pred, score in zip(predictions, scores):
        rescore_msg = create_rescore_msg(pred, score)
        cc = completion_coroutine(sys_msg, rescore_msg, model=model)
        coroutines.append(cc)
        sleep(0.05)
    # run the asynchronous gathering function
    return asyncio.run(gather_completions(coroutines, n_jobs=n_jobs))

# all numbers here from GAD-939 with 5000 beams
# 21.4%
sys_msg = """
Your task is automatic speech recognition. \
Below are the candidate transcriptions, ordered from most likely \
to least likely. \
Respond with the correct transcription, \
without any introductory text.
""".strip()

# 20.95% w/ GPT-4, 20.58% w/ GPT-3.5 [10 preds: 20.68% vs 20.58% for 100 preds]
sys_msg = "Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. Choose the transcription that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond with the chosen transcription only, without any introductory text."

# sys_msg = """Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. Choose the transcription that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond with the chosen transcription only, without any introductory text."""

# 22.11% with GPT-3.5, and 2 examples don't conform (10 examples)
# sys_msg = "Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. Begin your response with a Chain of Reasoning, explaining your analysis and decision-making process in choosing the most accurate transcription. After your analysis, clearly indicate your final choice with the cue 'TRANSCRIPT: '. Ensure the transcription you choose is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond first with your reasoning, followed by 'TRANSCRIPT: ' and then the chosen transcription."

# 26.13% with GPT-3.5, and 1 example doesn't conform (10 examples)
# sys_msg = "Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. Choose the three transcriptions that are most accurate, ensuring they are contextually and grammatically correct. Briefly explain your reasoning on which to choose in one or two sentences, focusing on key differences in the options that influence your decision. After your brief reasoning, clearly indicate your final choice with the cue 'TRANSCRIPT: '. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond first with your concise reasoning, followed by 'TRANSCRIPT: ' and then the chosen transcription."

# GPT-3.5 21.29%
# sys_msg = "Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. If the correct transcription is not explicitly listed, use your understanding of context and language to infer the most likely correct wording. Choose the transcription (or inferred correct version) that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases where the correct transcription is not clear, select or infer the option that is most coherent and contextually sound. Respond with the chosen or inferred transcription only, without any introductory text."

def create_rescore_msg(predictions, scores):
    rescore_msg = "\n".join([p for p in predictions])
    return rescore_msg 

# def create_rescore_msg(predictions, scores):
#     rescore_msg = "\n".join([f"{s:.3f}\t{p}" for p, s in zip(predictions, scores)])
#     return rescore_msg  

# model = "gpt-4-0125-preview" # 21.85% took 4min
# model = "gpt-3.5-turbo-16k-0613" # 21.42 - 21.45%, took 2:10 - 2:37 with 3 jobs
model = "gpt-3.5-turbo-1106" # 21.00% took 6:41 with 3 jobs (bad timeout..?)
# Call batch_completions
lisa_predictions = batch_completions(silent_predictions, silent_beam_scores, sys_msg,
                                     model=model)
##
# def clean_transcripts(transcripts):
#     ret = []
#     for transcript in transcripts:
#         # split on 'TRANSCRIPT: '
#         t = transcript.split("TRANSCRIPT: ")[-1]
#         # remove leading and trailing whitespace
#         t = t.strip()
#         ret.append(t)
#     ret = list(map(text_transform.clean_text, ret))
#     transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
#     ret = transformation(ret)
#     return ret
lisa_wer = calc_wer(clean_transcripts(lisa_predictions), silent_labels, text_transform)
typer.echo(f"Baseline WER: {baseline_wer * 100:.2f}%")  # repeat due to noisy output
typer.echo(f"Final WER with {model}: {lisa_wer * 100:.2f}%")
##
c.choices[0].message.content
##
# Get transcripts using the provided function
transcripts = batch_predict_from_topk(
    silent_predictions, silent_beam_scores, sys_msg=sys_msg, n_jobs=n_jobs
)

# Clean and calculate WER for the transcripts

typer.echo(f"Baseline WER: {baseline_wer * 100:.2f}%")  # repeat due to noisy output


##
for i,t in enumerate(lisa_predictions):
    if not "TRANSCRIPT: " in t:
        print(i,t)
##
# nondestructively drop 34 and 167 on new copy
new_preds = np.copy(lisa_predictions)
new_preds = np.delete(new_preds, 197)
# new_preds = np.delete(new_preds, 34)
new_labels = np.copy(silent_labels)
new_labels = np.delete(new_labels, 197)
# new_labels = np.delete(new_labels, 34)
lisa_wer = calc_wer(clean_transcripts(new_preds), new_labels, text_transform)
typer.echo(f"Baseline WER: {baseline_wer * 100:.2f}%")  # repeat due to noisy output
typer.echo(f"Final WER with {model}: {lisa_wer * 100:.2f}%")
##
