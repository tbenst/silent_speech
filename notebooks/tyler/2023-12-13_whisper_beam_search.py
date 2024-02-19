##
# !pip install transformers optimum accelerate evaluate jiwer librosa soundfile
##
from transformers import pipeline
import torch, os, numpy as np
model_name = "openai/whisper-large-v3"

pipe = pipeline("automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch.float16,
                device="cuda:0",
                # TODO install CUDA 11.6
                # model_kwargs={"use_flash_attention_2": True}
)

# pipe.model = pipe.model.to_bettertransformer()

# Process the audio file
base_dir = "/scratch/GaddyPaper/emg_data/nonparallel_data/4-24/"
i = 413

audio_file = os.path.join(base_dir, f"{i}_audio.wav")
##
# preprocess audio
# https://github.com/huggingface/transformers/blob/131a528be02e1fa2d27f215920d2fd69e1d246cd/src/transformers/pipelines/base.py#L1077
model_inputs = next(pipe.preprocess(audio_file))
model_outputs = pipe.forward(model_inputs)

##
k = 20
outputs = pipe(audio_file,
    chunk_length_s=30,
    batch_size=45,
    return_timestamps=True,
    generate_kwargs={"language": "en", "task": "transcribe",
                     "do_sample": True,
                     "num_beams": k, "num_return_sequences": k,
                    #  "return_dict_in_generate": True,
                     "output_scores": True},
)
outputs
##
import json, evaluate
wer = evaluate.load("wer")
file = os.path.join(base_dir, f"{i}_info.json")
with open(file, "r") as f:
    info = json.load(f)
text = info['text']
print("Number of words in text:", len(text.split()))
e = wer.compute(predictions=[text], references=[outputs['text']])
print("WER:", e)

##
# hard to get multiple beams from whisper that are different and don't suck
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Select an audio file and read it:
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = ds[0]["audio"]
waveform = audio_sample["array"]
sampling_rate = audio_sample["sampling_rate"]

# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")

# Use the model and processor to transcribe the audio:
input_features = processor(
    waveform, sampling_rate=sampling_rate, return_tensors="pt"
).input_features.to("cuda")

num_beams = 10
# Generate token ids with beam search
predicted_output = model.generate(
    input_features,
    num_beams=num_beams,  # Number of beams for beam search
    num_beam_groups=num_beams//2,
    # whisper small: 0.5 too small, 5.0 is rambly
    # whisper v3: 1.0 too small, 5.0 rambles
    diversity_penalty=10.0,
    num_return_sequences=num_beams,  # Number of sequences to return
    output_scores=True,
    return_dict_in_generate=True,
    length_penalty=-1.0,
    language="en", task="transcribe",
    # do_sample=True,
    early_stopping=True  # Stop the beam search when the first beam is finished
)

sequences = predicted_output.sequences
scores = predicted_output.sequences_scores

# Decode sequences to text
transcriptions = processor.batch_decode(sequences, skip_special_tokens=True)

# Print each transcription along with its score
for i, (transcription, score) in enumerate(zip(transcriptions, scores)):
    print(f"Score: {score}: {transcription}")


##
