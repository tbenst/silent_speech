##
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4Tv2ForSpeechToText
import os, torchaudio
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to("cuda")
# model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large").to("cuda")

##
# from datasets import load_dataset
# dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)
# audio_sample = next(iter(dataset))["audio"]
# audio_sample['array'].shape
##
base_dir = "/scratch/GaddyPaper/emg_data/nonparallel_data/4-24/"
i = 413
audio_file = os.path.join(base_dir, f"{i}_audio.wav")
# read wav
audio = torchaudio.load(audio_file)[0].squeeze()
audio_inputs = processor(audios=audio, return_tensors="pt",
                         sampling_rate=16000)
audio_inputs = {k: v.cuda() for k, v in audio_inputs.items()}
n_reps = 10
audio_inputs['input_features'] = audio_inputs['input_features'].repeat(n_reps, 1, 1)
audio_inputs['attention_mask'] = audio_inputs['attention_mask'].repeat(n_reps, 1)
##
num_beams = 10
output_tokens = model.generate(**audio_inputs,
    tgt_lang="eng", generate_speech=False,
    text_do_sample=True, text_temperature=1.5
    )
translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
translated_text_from_audio
##
# !pip install praat-textgrids
import json, evaluate, sys
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from data_utils import TextTransform

text_transform = TextTransform()
wer = evaluate.load("wer")
file = os.path.join(base_dir, f"{i}_info.json")
with open(file, "r") as f:
    info = json.load(f)
text = info['text']
print("Number of words in text:", len(text.split()))

e = wer.compute(
    predictions=[text_transform.clean_text(translated_text_from_audio)],
    references=[text_transform.clean_text(text)])
print("WER:", e)
##
