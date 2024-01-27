import re, os, numpy as np, torch
from tqdm import tqdm
import torch.nn as nn
from data_utils import TextTransform
from architecture import Model, S4Model, H3Model, ResBlock, MONAConfig, MONA
from torchaudio.models.decoder import ctc_decoder
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import jiwer
import neptune.new as neptune
from pytorch_lightning.loggers import NeptuneLogger


def sentence_to_fn(sentence, directory, ext=".wav"):
    fn = re.sub(r"[^\w\s]", "", sentence)  # remove punctuation
    fn = fn.lower().replace(" ", "_")  # lowercase with underscores
    return os.path.join(directory, fn + ext)


def string_to_np_array(string):
    """
    Convert a string representation of a numpy array into an actual numpy array.
    """
    try:
        # Remove square brackets and split the string by spaces
        elements = string.strip("[]").split()
        # Convert each element to float and create a numpy array
        return np.array([float(element) for element in elements])
    except Exception as e:
        print(f"Error converting string to numpy array: {e}")
        return None


def load_npz_to_memory(npz_path, **kwargs):
    npz = np.load(npz_path, **kwargs)
    loaded_data = {k: npz[k] for k in npz}
    npz.close()
    return loaded_data


def load_model(ckpt_path, config):
    text_transform = TextTransform(togglePhones=config.togglePhones)
    model = MONA(config, text_transform, no_neural=True)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def get_emg_pred(model, dataloader):
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            X = nn.utils.rnn.pad_sequence(batch["raw_emg"], batch_first=True)
            X = X.cuda()
            pred = model.emg_forward(X)[0].cpu()
            predictions.append((batch, pred))
    return predictions


def get_audio_pred(model, dataloader):
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            X = nn.utils.rnn.pad_sequence(batch["audio_features"], batch_first=True)
            X = X.cuda()
            pred = model.audio_forward(X)[0].cpu()
            predictions.append((batch, pred))
    return predictions


# Function to run the beam search
def run_beam_search(
    batch_pred,
    text_transform,
    k,
    lm_weight,
    beam_size,
    beam_threshold,
    use_lm,
    togglePhones,
    lexicon_file,
    lm_file,
):
    batch, pred = batch_pred

    if use_lm:
        lm = lm_file
    else:
        lm = None

    decoder = ctc_decoder(
        lexicon=lexicon_file,
        tokens=text_transform.chars + ["_"],
        lm=lm,
        blank_token="_",
        sil_token="|",
        nbest=k,
        lm_weight=lm_weight,
        beam_size=beam_size,
        beam_threshold=beam_threshold,
    )

    beam_results = decoder(pred)
    all_trl_top_k = []
    all_trl_beam_scores = []
    all_sentences = []
    for i, (example, beam_result) in enumerate(zip(batch, beam_results)):
        # Filter out silences
        target_sentence = text_transform.clean_text(batch["text"][i])
        if len(target_sentence) > 0:
            trl_top_k = []
            trl_beam_scores = []
            for beam in beam_result:
                transcript = " ".join(beam.words).strip().lower()
                score = beam.score
                trl_top_k.append(transcript)
                trl_beam_scores.append(score)

            all_trl_top_k.append(np.array(trl_top_k))
            all_trl_beam_scores.append(np.array(trl_beam_scores))
            all_sentences.append(target_sentence)
        else:
            all_trl_top_k.append(np.array([]))
            all_trl_beam_scores.append(np.array([]))
            all_sentences.append(target_sentence)

    return all_trl_top_k, all_trl_beam_scores, all_sentences


def get_top_k(
    predictions,
    text_transform,
    k: int = 100,
    beam_size: int = 500,
    togglePhones: bool = False,
    use_lm: bool = True,
    beam_threshold: int = 100,
    lm_weight: float = 2,
    cpus: int = 8,
    lexicon_file: str = None,
    lm_file: str = None,
):
    # Define the function to be used with concurrent.futures
    func = partial(
        run_beam_search,
        text_transform=text_transform,
        k=k,
        lm_weight=lm_weight,
        beam_size=beam_size,
        beam_threshold=beam_threshold,
        use_lm=use_lm,
        togglePhones=togglePhones,
        lexicon_file=lexicon_file,
        lm_file=lm_file,
    )

    # If cpus=0, run without multiprocessing
    if cpus == 0:
        beam_results = [func(pred) for pred in tqdm(predictions)]
    else:
        # Use concurrent.futures for running the beam search with a progress bar
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            beam_results = list(
                tqdm(executor.map(func, predictions), total=len(predictions))
            )

    # flatten batched tuples of (all_trl_top_k, all_trl_beam_scores, all_sentences)
    # Separate and flatten the results
    all_trl_top_k, all_trl_beam_scores, all_sentences = [], [], []
    for trl_top_k, trl_beam_scores, sentences in beam_results:
        all_trl_top_k.extend(trl_top_k)
        all_trl_beam_scores.extend(trl_beam_scores)
        all_sentences.extend(sentences)

    # Collecting results
    topk_dict = {
        "k": k,
        "beam_size": beam_size,
        "beam_threshold": beam_threshold,
        "sentences": np.array(all_sentences),
        "predictions": np.array(all_trl_top_k, dtype=object),  # ragged array
        "beam_scores": np.array(all_trl_beam_scores, dtype=object),
    }

    return topk_dict


def calc_wer(predictions, targets, text_transform):
    """Calculate WER from predictions and targets.

    predictions: list of strings
    targets: list of strings
    """
    if type(predictions) is np.ndarray:
        predictions = list(map(str, predictions))
    if type(targets) is np.ndarray:
        targets = list(map(str, targets))
    targets = list(map(text_transform.clean_text, targets))
    predictions = list(map(text_transform.clean_text, predictions))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)
    # print(targets, predictions)
    return jiwer.wer(targets, predictions)


def get_neptune_run(run_id, mode="read-only", **neptune_kwargs):
    return NeptuneLogger(
        run=neptune.init_run(
            with_id=run_id,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            mode=mode,
            **neptune_kwargs,
        ),
        log_model_checkpoints=False,
    )


def get_best_ckpts(directory, n=1):
    # get all files ending in .ckpt in subdirectories of directory
    ckpt_paths = []
    metrics = []
    # extract wer from eg "silent_emg_wer=0.253.ckpt"
    r = re.compile("wer=(0\.\d+)")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ckpt"):
                try:
                    metrics.append(float(r.findall(file)[0]))
                    ckpt_paths.append(os.path.join(root, file))
                except IndexError:
                    pass
    perm = np.argsort(metrics)
    return [ckpt_paths[i] for i in perm[:n]], [metrics[i] for i in perm[:n]]


def get_last_ckpt(directory):
    """Get the most recent checkpoint for e.g. resuming a run."""
    ckpt_paths = []
    epochs = []

    # Regular expression to extract the epoch number from the directory name
    r = re.compile(r"epoch=(\d+)-")

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ckpt"):
                dir_name = os.path.basename(root)
                match = r.search(dir_name)
                if match:
                    epoch = int(match.group(1))
                    epochs.append(epoch)
                    ckpt_paths.append(os.path.join(root, file))

    # Find the index of the checkpoint with the highest epoch number
    if epochs:  # Ensure list is not empty
        max_epoch_idx = np.argmax(epochs)
        return ckpt_paths[max_epoch_idx], epochs[max_epoch_idx]
    else:
        raise ValueError("No checkpoints found in directory.")


def nep_get(logger, key):
    val_promise = logger.experiment.get_attribute(key)
    if hasattr(val_promise, "fetch"):
        return val_promise.fetch()
    elif hasattr(val_promise, "fetch_values"):
        return val_promise.fetch_values()
    else:
        raise NotImplementedError("don't know how to fetch values")


def load_model_from_id(run_id, choose="best"):
    assert choose in ["best", "last"]

    neptune_logger = NeptuneLogger(
        run=neptune.init_run(
            with_id=run_id,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            mode="read-only",
            project="neuro/Gaddy",
        ),
        log_model_checkpoints=False,
    )
    output_directory = nep_get(neptune_logger, "output_directory")
    hparams = nep_get(neptune_logger, "training/hyperparams")
    if choose == "best":
        ckpt_paths, wers = get_best_ckpts(output_directory, n=1)
        wer = wers[0]
        ckpt_path = ckpt_paths[0]
        min_wer = nep_get(neptune_logger, "training/val/wer").value.min()
        assert np.isclose(wer, min_wer, atol=1e-3), f"wer {wer} != min_wer {min_wer}"
        print("found checkpoint with WER", wer)
    elif choose == "last":
        ckpt_path, epoch = get_last_ckpt(output_directory)
        assert (
            epoch == hparams["max_epochs"]
        ), f"epoch {epoch} != max_epochs {hparams['max_epochs']}"
        print("found checkpoint with epoch", epoch)
    togglePhones = hparams["togglePhones"]
    assert togglePhones == False, "not implemented"
    if "use_supCon" in hparams:
        hparams["use_supTcon"] = hparams["use_supCon"]
        del hparams["use_supCon"]
    config = MONAConfig(**hparams)

    model = load_model(ckpt_path, config)
    return model, config, output_directory
