import whisper
import os
import tqdm
import soundfile as sf
import librosa
import jiwer
import logging
from unidecode import unidecode

def evaluate(testset, audio_directory):
    model = whisper.load_model("large")
    predictions = []
    targets = []

    for i, datapoint in enumerate(tqdm.tqdm(testset, 'Evaluate outputs', disable=None)):
        audio_path = os.path.join(audio_directory, f'example_output_{i}.wav')
        audio, rate = sf.read(audio_path)

        if rate != 16000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
        result = model.transcribe(audio_path)
        text = result["text"]

        predictions.append(text)
        target_text = unidecode(datapoint['text'])
        targets.append(target_text)

    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)

    logging.info(f'targets: {targets}')
    logging.info(f'predictions: {predictions}')
    # logging.info(f'wer: {jiwer.wer(targets, predictions)}')

    filtered_targets = []
    filtered_predictions = []
    for target, prediction in zip(targets, predictions):
        if target.strip(): 
            filtered_targets.append(target)
            filtered_predictions.append(prediction)

    if len(filtered_targets) == 0:
        logging.warning("All reference transcriptions are empty! WER cannot be computed.")
    else:
        logging.info(f'wer: {jiwer.wer(filtered_targets, filtered_predictions)}')


