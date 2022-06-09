import argparse
import os
from pathlib import Path
from encoder.audio import trim_long_silences

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder


def voice_change(text, file_path, encoder_path = 'saved_models\\default\\encoder.pt', synth_path = 'saved_models\\default\\synthesizer.pt', vocoder_path = 'saved_models\\default\\vocoder.pt', trim_silence = True, seed = None, cpu=False, no_sound=False):

    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ensure_default_models(Path("saved_models"))
    encoder.load_model(encoder_path)
    synthesizer = Synthesizer(synth_path)
    vocoder.load_model(vocoder_path, verbose=False)

    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    mels = synthesizer.synthesize_spectrograms(texts, embeds)
    mel = np.concatenate(mels, axis=1)
    no_action = lambda *args: None

    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

    num_generated = 0
    a = True
    while a:
        try:
            # Get the reference audio filepath
            message = file_path
            in_fpath = Path(message.replace("\"", "").replace("\'", ""))

            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            embed = encoder.embed_utterance(preprocessed_wav)


            
            # text = input("Write a sentence (+-20 words) to be synthesized:\n")
            if seed is not None:
                torch.t(torch.tensor([seed]))
                synthesizer = Synthesizer(synth_path)

            texts = [text]
            embeds = [embed]
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]

            if seed is not None:
                torch.manual_seed(seed)
                vocoder.load_model(vocoder_path, verbose=False)
            generated_wav = vocoder.infer_waveform(spec)
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
            generated_wav = encoder.preprocess_wav(generated_wav)
            # if not no_sound:
            #     import sounddevice as sd
            #     try:
            #         sd.stop()
            #         sd.play(generated_wav, synthesizer.sample_rate)
            #     except sd.PortAudioError as e:
            #         print("\nCaught exception: %s" % repr(e))
            #         print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
            #     except:
            #         raise
            # print('cv')
            print(file_path)
            filename = f"{file_path}_generated.wav"
            # print('filename d')
            # if trim_silence:
            #    trim_long_silences(generated_wav)
            # sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            # print('write d')
            num_generated += 1
            # print("\nSaved output as %s\n\n" % filename)


        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
            
        finally:
            a = False
            return generated_wav

if __name__=="__main__":
    voice_change("You should suck some goat dick, it's very tasty.", 'data/trump11.wav', seed= 0)
