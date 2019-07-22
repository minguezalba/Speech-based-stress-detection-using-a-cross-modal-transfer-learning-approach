from utils.files_manager import read_directory
from utils.audio_utils import AudioManager
from utils.labels_utils import LabelsManager
from utils.image_utils import ImageManager

DIR_AUDIOS = 'data/processed_audios/'
EXT_AUDIOS = 'wav'
DIR_LABELS = 'data/labels/'
EXT_LABELS = 'csv'


def generate_images(method_image):

    audio_files = read_directory(DIR_AUDIOS, EXT_AUDIOS)
    label_files = read_directory(DIR_LABELS, EXT_LABELS)

    for (f_path, f_name), (l_path, l_name) in zip(audio_files, label_files):

        print(f'Loading audio file {f_name}')
        audio_manager = AudioManager.load_audio(f_path, f_name)
        labels_manager = LabelsManager.load_labels(l_path, l_name)

        for i, audio_frame, label in enumerate(zip(audio_manager.data, labels_manager.data)):

            image_manager = ImageManager.generate_image(method_image, audio_frame, i, f_name)
            image_manager.save_image(method_image, label, show=True)

            print(1)


if __name__ == '__main__':
    method_image_ = 'RGB'  # RBG or greyscale
    generate_images(method_image_)