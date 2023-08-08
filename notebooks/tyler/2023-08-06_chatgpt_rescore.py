##
import numpy as np
import sys, os
import openai, jiwer
import tiktoken
from scipy.io import loadmat
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from data_utils import TextTransform
from pqdm.threads import pqdm
import backoff  # for exponential backoff

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_value=15, max_time=90)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
text_transform = TextTransform(togglePhones = False)

# npz = np.load("/scratch/users/tbenst/2023-08-01T06:54:28.359594_gaddy/SpeechOrEMGToText-epoch=199-val/top100_500beams.npz",
#               allow_pickle=True)
npz = np.load("/scratch/users/tbenst/2023-08-01T06:54:28.359594_gaddy/SpeechOrEMGToText-epoch=199-val/top100_5000beams.npz",
              allow_pickle=True)
##
sys_msg = """
You are a rescoring algorithm for automatic speech recognition. \
Given the results of a beam search, with candidate hypotheses and their score, \
respond with the correct transcription.
""".strip()

def create_rescore_msg(predictions, scores):
    rescore_msg = '\n'.join([f"{s:.3f}\t{p}"
        for p,s in zip(predictions, scores)])
    return rescore_msg

def predict_from_topk(predictions, scores, sys_msg=sys_msg):
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
    
    cc = completions_with_backoff(
        model=model,
        messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": rescore_msg},
            ]
    )
    
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

def batch_predict_from_topk(predictions, scores, sys_msg=sys_msg):
    pt = lambda x: predict_from_topk(*x, sys_msg=sys_msg)
    # can only request up to 90k tokens / minute
    # njobs = 16 # rate limited
    # njobs = 4 # 2:28
    njobs = 3 # 1:31, sometimes 10min tho..?
    # njobs = 2 # 3:06, also sometimes 10min
    transcripts = pqdm(zip(predictions, scores), pt, n_jobs=njobs)
    for i,t in enumerate(transcripts):
        if type(t) != str:
            print(i,t)
            p = predict_from_topk(npz['predictions'][i], npz['beam_scores'][i])
            transcripts[i] = p
    return transcripts
    # return [predict_from_topk(pred, score, sys_msg)
    #     for pred, score in tqdm(zip(predictions, scores))]
    
def clean_transcripts(transcripts):
    transcripts = list(map(text_transform.clean_2, transcripts))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    transcripts     = transformation(transcripts)
    for i in range(len(transcripts)):
        if transcripts[i].startswith("the correct transcription is "):
            transcripts[i] = transcripts[i].replace("the correct transcription is ", "")
            
    return transcripts
##
# baseline (25.5% for 5000 beams!)
calc_wer([n[0] for n in npz['predictions']], npz['sentences'])
##
# transcript = predict_from_topk(npz['predictions'][0], npz['beam_scores'][0])
# transcript = create_rescore_msg(npz['predictions'][0], npz['beam_scores'][0])

transcripts = batch_predict_from_topk(npz['predictions'], npz['beam_scores'])
# 24.7% on 500 beams
# 24.3% on 5000 beams
calc_wer(clean_transcripts(transcripts), npz['sentences'])
##
calc_wer([p[0] for p in npz['predictions']], npz['sentences']) # 26.0%
##
sys_msg2 = "You are a rescoring algorithm for automatic speech recognition, focusing on generating coherent and contextually relevant transcriptions. Given a list of candidate transcriptions with scores produced by a beam search, your task is to deduce the most likely transcription that makes sense contextually and grammatically, even if it's not explicitly present in the given options."
transcripts = batch_predict_from_topk(npz['predictions'], npz['beam_scores'], sys_msg=sys_msg2)
calc_wer(clean_transcripts(transcripts), npz['sentences']) # > 100%
##
sys_msg2 = "You are a rescoring algorithm for automatic speech recognition, focusing on generating coherent and contextually relevant transcriptions. Given a list of candidate transcriptions with scores produced by a beam search, your task is to deduce the most likely transcription that makes sense contextually and grammatically from the given options. Your response should be the corrected transcription only, without any additional explanation or introductory text. Make only minimal changes to correct grammar and coherence, and ensure that the transcription is closely based on one of the provided candidates."
transcripts = batch_predict_from_topk(npz['predictions'], npz['beam_scores'], sys_msg=sys_msg2)
calc_wer(clean_transcripts(transcripts), npz['sentences']) # 26.3%
##
sys_msg2 = "You are a rescoring algorithm for automatic speech recognition, focusing on generating coherent and contextually relevant transcriptions. Given a list of candidate transcriptions with scores produced by a beam search, your task is to deduce the most likely transcription that makes sense contextually and grammatically, even if it's not explicitly present in the given options. Your response should be the corrected transcription only, without any additional explanation or introductory text."
transcripts = batch_predict_from_topk(npz['predictions'], npz['beam_scores'], sys_msg=sys_msg2)
calc_wer(clean_transcripts(transcripts), npz['sentences']) # 24.49%
##
sys_msg2 = """
Your task is automatic speech recognition. \
Below are the candidate transcriptions along with their \
negative log-likelihood from a CTC beam search. \
Respond with the correct transcription, \
without any introductory text.
""".strip()
transcripts = batch_predict_from_topk(npz['predictions'], npz['beam_scores'], sys_msg=sys_msg2)
calc_wer(clean_transcripts(transcripts), npz['sentences']) # 22.85% !!!!
##

for t, p, s in zip(clean_transcripts(transcripts), [p[0] for p in npz['predictions']], npz['sentences']):
    if t != p:
        print("===============")
        print(s)
        print(p)
        print(t)
##
