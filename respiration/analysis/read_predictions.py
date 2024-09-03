import numpy as np
import pandas as pd
import respiration.utils as utils

from typing import Optional


class PredictionsReader:
    # Path to the directory containing the predictions
    path: str

    def __init__(self, path: Optional[str] = None):
        if path is None:
            self.path = utils.dir_path('outputs', 'signals')
        else:
            self.path = path

    def __read_optical_flow(self, file: str) -> pd.DataFrame:
        filepath = utils.join_paths(self.path, file)
        predictions = pd.read_csv(filepath)
        predictions['signal'] = predictions['signal_v'].apply(eval).apply(np.array)

        # Only keep the chest roi predictions
        predictions = predictions[predictions['roi'] == 'chest']

        # Only keep the columns that are needed
        return predictions[['subject', 'setting', 'model', 'signal']]

    def read_raft(self, file: str = 'raft_predictions.csv') -> pd.DataFrame:
        return self.__read_optical_flow(file)

    def read_flownet(self, file: str = 'flownet_predictions.csv') -> pd.DataFrame:
        return self.__read_optical_flow(file)

    def read_pretrained(self, file: str = 'pretrained_predictions.csv') -> pd.DataFrame:
        pretrained_file = utils.join_paths(self.path, file)
        pretrained_predictions = pd.read_csv(pretrained_file)
        pretrained_predictions['signal'] = pretrained_predictions['signal'].apply(eval).apply(np.array)

        # Only keep the columns that are needed
        return pretrained_predictions[['subject', 'setting', 'model', 'signal']]

    def read_lucas_kanade(self, file: str = 'lucas_kanade.csv') -> pd.DataFrame:
        lucas_kanade_file = utils.join_paths(self.path, file)
        lucas_kanade = pd.read_csv(lucas_kanade_file)
        lucas_kanade['signal'] = lucas_kanade['signal'].apply(eval).apply(np.array)

        # Rename column method to model
        lucas_kanade.rename(columns={'method': 'model'}, inplace=True)

        # Remove all the rows that have a signal with a length of 0
        lucas_kanade = lucas_kanade[lucas_kanade['grey'] == False]

        # Only keep the columns that are needed
        return lucas_kanade[['subject', 'setting', 'model', 'signal']]

    def read_pixel_intensity(self, file: str = 'pixel_intensity.csv') -> pd.DataFrame:
        pixel_intensity_file = utils.join_paths(self.path, 'pixel_intensity.csv')
        pixel_intensity = pd.read_csv(pixel_intensity_file)
        pixel_intensity['signal'] = pixel_intensity['signal'].apply(eval).apply(np.array)

        # Rename column method to model
        pixel_intensity.rename(columns={'method': 'model'}, inplace=True)

        # Only keep the columns that are needed
        return pixel_intensity[['subject', 'setting', 'model', 'signal']]

    def read_rppg(self, file: str = 'r_ppg_predictions.csv') -> pd.DataFrame:
        r_ppg_path = utils.join_paths(self.path, file)

        r_ppg_prediction = pd.read_csv(r_ppg_path)
        r_ppg_prediction['signal'] = r_ppg_prediction['signal'].apply(eval).apply(np.array)

        # Only keep the columns that are needed
        return r_ppg_prediction[['subject', 'setting', 'model', 'signal']]

    def read_simple_vit(self, file: str = 'transformer_predictions.csv') -> pd.DataFrame:
        transformer_path = utils.join_paths(self.path, file)

        transformer_prediction = pd.read_csv(transformer_path)
        transformer_prediction['signal'] = transformer_prediction['signal'].apply(eval).apply(np.array)

        # Add a tf_ prefix to the model names
        transformer_prediction['model'] = 'SimpleViT_' + transformer_prediction['model']

        # Only keep the columns that are needed
        return transformer_prediction[['subject', 'setting', 'model', 'signal']]

    def read_random(self, file: str = 'random_predictions.csv') -> pd.DataFrame:
        random_path = utils.join_paths(self.path, file)

        random_prediction = pd.read_csv(random_path)
        random_prediction['signal'] = random_prediction['signal'].apply(eval).apply(np.array)

        # Only keep the columns that are needed
        return random_prediction[['subject', 'setting', 'model', 'signal']]

    def read_rhythm_former(self, file: str = 'rhythm_former.csv') -> pd.DataFrame:
        rhythm_former_path = utils.join_paths(self.path, 'rhythm_former.csv')

        rhythm_former = pd.read_csv(rhythm_former_path)
        rhythm_former['signal'] = rhythm_former['signal'].apply(eval).apply(np.array)

        # Only keep the columns that are needed
        return rhythm_former[['subject', 'setting', 'model', 'signal']]

    def read_efficient_phys(self, file: str = 'efficient_phys_predictions.csv') -> pd.DataFrame:
        efficient_phys_path = utils.join_paths(self.path, file)

        efficient_phys_predictions = pd.read_csv(efficient_phys_path)
        efficient_phys_predictions['signal'] = efficient_phys_predictions['signal'].apply(eval).apply(np.array)

        # Only keep the columns that are needed
        return efficient_phys_predictions[['subject', 'setting', 'model', 'signal']]

    def read_all(self) -> pd.DataFrame:
        return pd.concat([
            self.read_raft(),
            self.read_flownet(),
            self.read_pretrained(),
            self.read_lucas_kanade(),
            self.read_pixel_intensity(),
            self.read_rppg(),
            self.read_simple_vit(),
            self.read_random(),
            self.read_rhythm_former(),
            self.read_efficient_phys(),
        ])
