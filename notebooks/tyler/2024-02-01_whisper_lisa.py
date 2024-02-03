##
import os, sys, json
from openai import OpenAI
from glob import glob
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from read_emg import TESTSET_FILE
from data_utils import TextTransform
from helpers import calc_wer

with open(TESTSET_FILE) as f:
    testset_json = json.load(f)
    valset = testset_json["dev"]
    testset = testset_json["test"]
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
##
# print(os.environ["OPENAI_API_KEY"])
client = OpenAI()

pred_text = []
for wav_file in tqdm(val_wav_files):
    with open(wav_file, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        pred_text.append(transcript.text)
##
print(f"target: {val_text[0]}\ntranscript: {pred_text[0]}")
##
text_transform = TextTransform(togglePhones=False)
##
for i, s in enumerate(val_text):
    if text_transform.clean_text(s) == "":
        print(i)
##
# drop idx 165. 199 sentences in val set
val_text_nonempty = val_text[:165] + val_text[166:]
pred_text_nonempty = pred_text[:165] + pred_text[166:]
wer = calc_wer(pred_text_nonempty, val_text_nonempty, text_transform)
print(f"Whisper Validation WER: {wer * 100:.2f}%")  # 2.62%
##
# 99 sentences
test_pred_text = []
for wav_file in tqdm(test_wav_files):
    with open(wav_file, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        test_pred_text.append(transcript.text)

for i, s in enumerate(test_text):
    if text_transform.clean_text(s) == "":
        print(i)
##
bad_idx = 32
test_text_nonempty = test_text[:bad_idx] + test_text[bad_idx + 1 :]
pred_text_nonempty = test_pred_text[:bad_idx] + test_pred_text[bad_idx + 1 :]
wer = calc_wer(pred_text_nonempty, test_text_nonempty, text_transform)
print(f"Whisper Test WER: {wer * 100:.2f}%")  # 2.31%
##
