from dataclasses import dataclass

import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.audio_utils import SAMPLE_RATE

DIR_IMAGES = 'data/images/'
EXT_IMAGES = 'png'


N_MELS = 126  # number of Mel bands to generate
N_FFT = 512 # freq resolution points
HOP_COEF = 0.25  # hop coefficient
HOP_LENGTH = int(N_FFT*HOP_COEF)
F_MIN = 50  # hertz
F_MAX = SAMPLE_RATE/2


@dataclass
class ImageManager:
    path: str
    filename: str
    method: str
    values: np.ndarray

    @staticmethod
    def generate_image(method, audio_frame, i, filename, label):
        filename = f'{filename}_{i}.{EXT_IMAGES}'
        path = f'{DIR_IMAGES}{method}/{label}/{filename}'
        data = ImageManager._melspectrogram_operation(audio_frame)

        return ImageManager(path, filename, method, data)

    @staticmethod
    def _melspectrogram_operation(audio_frame):
        mel_spec = librosa.feature.melspectrogram(audio_frame, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                  n_mels=N_MELS, sr=SAMPLE_RATE, power=1.0,
                                                  fmin=F_MIN, fmax=F_MAX)

        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def save_image(self, method, label, show=False):

        # Normalization between (0,1) range
        x_max, x_min = self.values.max(), self.values.min()
        data_norm = (self.values - x_min) / (x_max - x_min)

        if method == 'RGB':
            three_channels_image = Image.fromarray(np.uint8(plt.cm.inferno(data_norm) * 255)).convert('RGB')
            three_channels_image.save(self.path)

            if show:
                three_channels_image.show()

        elif method == 'greyscale':
            three_channels_image = np.stack((data_norm, data_norm, data_norm), axis=2)
            plt.imshow(three_channels_image)
            plt.axis('off')
            plt.savefig(self.path, bbox_inches='tight', pad_inches=0)
            plt.close()

            if show:
                plt.show()

        else:
            print('Error: not available saving image method.')

    def show_plot(self, as_plot=False, im=None):

        if as_plot:
            w, h = plt.figaspect(self.values.shape[0] / self.values.shape[1])
            fig = plt.figure(figsize=(w, h))

            librosa.display.specshow(self.values, x_axis='ms', y_axis='mel',
                                     sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                     fmin=F_MIN, fmax=F_MAX)

            title = 'n_mels={}, f_min={}, f_max={} time_steps={}, fft_bins={}  (2D resulting shape: {})'
            plt.title(title.format(N_MELS, F_MIN, F_MAX, self.values.shape[1], self.values.shape[0], self.values.shape))
        else:
            im.show()
