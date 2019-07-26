from dataclasses import dataclass

import librosa
import matplotlib.pyplot as plt
import numpy as np

DIR_AUDIOS = '../data/processed_audios/'

SAMPLE_RATE = 16000  # hertz
WIN = 1*SAMPLE_RATE  # samples



@dataclass
class AudioManager:
    path_in: str
    filename: str
    data: np.ndarray
    n_frames: int
    fs: int = SAMPLE_RATE
    win: int = WIN

    @staticmethod
    def load_audio(file_path, file_name):
        audio_frames = np.empty([0, WIN], dtype=float)

        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        n_frames = int(np.floor(len(audio)/SAMPLE_RATE))

        for n in range(n_frames):
            frame = audio[n * WIN : (n + 1) * WIN]
            audio_frames = np.vstack([audio_frames, frame]) if audio_frames.size else frame

        if audio_frames.ndim == 1:
            audio_frames = audio_frames[np.newaxis, :]

        return AudioManager(file_path, file_name, audio_frames, audio_frames.shape[0])

    @staticmethod
    def show_audio(audio_samples, fs, filename):
        plt.figure()
        time_steps = np.arange(len(audio_samples)) * 100.0 / fs  # in miliseconds
        plt.plot(time_steps, audio_samples)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.title(f'{filename}')
        plt.show()
        # plt.close()
