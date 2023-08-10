import numpy as np
import sys, os
import openai, jiwer, typer
import tiktoken
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from data_utils import TextTransform
from pqdm.threads import pqdm

import logging

logging.basicConfig(level=logging.INFO)

# can use tenacity or backoff
# https://platform.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


# TODO: still hangs on last example or two. Maybe rebase to:
# https://github.com/openai/openai-cookbook/blob/c651bfdda64ac049747c2a174cde1c946e2baf1d/examples/api_request_parallel_processor.py
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completions_with_backoff(**kwargs):
    try:
        return openai.ChatCompletion.create(**kwargs)
    except Exception as e:
        # Log the exception and any relevant information
        logging.error(f"An error occurred while making the API call: {str(e)}")
        # Optionally, you can log the full traceback for detailed debugging
        logging.exception("Full traceback:")
        # Re-raise the exception to allow the retry decorator to handle it
        raise


tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
text_transform = TextTransform(togglePhones = False)

app = typer.Typer()


def create_rescore_msg(predictions, scores):
    rescore_msg = '\n'.join([f"{s:.3f}\t{p}"
        for p,s in zip(predictions, scores)])
    return rescore_msg

def predict_from_topk(predictions, scores, sys_msg, index=None):
    """Use OpenAI's chat API to predict from topk beam search results.
    
    GPT-3-turbo (4k or 8k)
    cost: $0.0015 or $0.003 / 1K tokens
    avg tokens per call: 1500
    cost per validation loop (200 examples): $0.45 - $0.9
    
    GPT-4 (8k)
    cost: $0.03 / 1K tokens
    cost per validation loop: $9
    """
    rescore_msg = create_rescore_msg(predictions, scores)
    num_tokens = num_tokens_from_string(rescore_msg) + num_tokens_from_string(sys_msg)
    if num_tokens > 4096:
        model = "gpt-3.5-turbo-16k"
    else:
        model = "gpt-3.5-turbo"
    # model="gpt-4"

    num_pred = len(predictions)
    if num_pred < 10:
        print(f"WARNING: only {num_pred} predictions from beam search")
    
    logging.info(f"API call {index} for {model}, Number of tokens: {num_tokens}")
    cc = completions_with_backoff(
        model=model,
        messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": rescore_msg},
            ],
        temperature=0.0
    )
    logging.info(f"Finished API call {index}")
    
    return cc.choices[0].message.content

def num_tokens_from_string(string: str, tokenizer=tokenizer) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(tokenizer.encode(string))
    return num_tokens

def calc_wer(predictions, targets):
    """Calculate WER from predictions and targets.
    
    predictions: list of strings
    targets: list of strings
    """
    # print(targets, predictions)
    targets = list(map(text_transform.clean_2, targets))
    predictions = list(map(text_transform.clean_2, predictions))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets     = transformation(targets)
    predictions = transformation(predictions)
    # print(targets, predictions)
    return jiwer.wer(targets, predictions)

# Update the batch_predict_from_topk function to accept n_jobs as a parameter
def batch_predict_from_topk(predictions, scores, sys_msg, n_jobs=3):
    pt = lambda x: predict_from_topk(x[1][0], x[1][1], sys_msg=sys_msg, index=x[0])
    transcripts = pqdm(enumerate(zip(predictions, scores)), pt, n_jobs=n_jobs)  # Added enumerate to get index
 
    for i,t in enumerate(transcripts):
        if type(t) != str:
            print(i,t)
            p = predict_from_topk(predictions[i], scores[i], sys_msg=sys_msg, index=i)
            transcripts[i] = p
    return transcripts
    
def clean_transcripts(transcripts):
    transcripts = list(map(text_transform.clean_2, transcripts))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    transcripts     = transformation(transcripts)
    for i in range(len(transcripts)):
        if transcripts[i].startswith("the correct transcription is "):
            transcripts[i] = transcripts[i].replace("the correct transcription is ", "")
            
    return transcripts

@app.command()
def main(
    npz_file: str = typer.Argument(..., help="Path to the .npz file containing the predictions and scores"),
    sys_msg: str = typer.Option(None, help="System message to use as input (optional)"),
    n_jobs: int = typer.Option(3, help="Number of jobs for parallel processing"),
    baseline: bool = typer.Option(False, help="Calculate only the baseline WER"),  # Added this option
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
    baseline_wer = calc_wer([n[0] for n in npz['predictions']], npz['sentences'])
    typer.echo(f"Baseline WER: {baseline_wer * 100:.2f}%")

    if not baseline:
        # Get transcripts using the provided function
        transcripts = batch_predict_from_topk(npz['predictions'], npz['beam_scores'], sys_msg=sys_msg, n_jobs=n_jobs)

        # Clean and calculate WER for the transcripts
        wer = calc_wer(clean_transcripts(transcripts), npz['sentences'])
        typer.echo(f"Baseline WER: {baseline_wer * 100:.2f}%") # repeat due to noisy output
        typer.echo(f"Final WER: {wer * 100:.2f}%")


# Run the application
if __name__ == "__main__":
    app()
