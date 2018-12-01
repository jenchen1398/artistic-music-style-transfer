
import librosa as libr
import numpy as np
import torch
import os
# os.chdir('artistic-music-style-transfer/pytorch')
import torch.utils.data
import utils

def mu_law_encode(x, mu_quantization=256):
    """ Numpy implementation of mu-law encoding"""
    assert(x.max() <= 1.0)
    assert(x.min() >= -1.0)
    mu = mu_quantization - 1.
    scaling = np.log1p(mu)
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / scaling
    encoding = ((x_mu + 1) / 2 * mu + 0.5).long()
    return encoding
  
def mu_law_decode(x, mu_quantization=256):
    """ Numpy implementation of mu-law decoding"""
    assert(np.max(x) <= mu_quantization)
    assert(np.min(x) >= 0)
    x = x.float()
    mu = mu_quantization - 1.
    # Map values back to [-1, 1].
    signal = 2 * (x / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude
  
class MusicDataset(torch.utils.data.Dataset):
    """Music"""

    def __init__(self, root_dir, sr = 22050, clip_length = 1, range = 0.5):
        """
        Args:
            root_dir (string): Directory with all the music.
            sr (int): Sampling rate (all music will be resampled to this rate by default. Default = 22050)
            clip_length (float): Clip length in seconds
        """
        self.root_dir = root_dir
        self.sr = sr
        self.clip_length = clip_length
        self.range = range

        allowed_formats = ['.m4a', '.wav', '.mp3']

        data = []

        for file in os.listdir(self.root_dir):
            print(file)
            if not any((file.endswith(ext) for ext in allowed_formats)):
                continue

            try:
                X, sr = libr.load("{}/{}".format(root_dir, file), self.sr)
                assert(sr == self.sr)
                Y = libr.util.frame(X, self.sr * self.clip_length) # split into 1 second clips
                # TODO may need to batch these, remove random later
                Z = []
                for clip in Y:
                    Z.append(self.augment_pitch(clip))
                Y = Z
                data.append(Y)
                print("successfully loaded {} {}-second ({} sample) clip(s) from {}".format(len(Y), self.clip_length, self.clip_length * self.sr, file))
            except AssertionError as e:
                print("unable to load {}".format(file))
        if len(data) > 1:
          self.data = np.concatenate(data, axis = 1).T 

        # to speed this up, maybe something like this, i.e. augment first

#         pitch = np.random.random_sample(self.data.shape[1]) - 0.5 # how much to raise/lower by
#         dur = np.random.random_sample(self.data.shape[1]) / 4 + 0.25 # duration of subsample between [0.25, .5]
#         low = min(np.random.random_sample(self.data.shape[1]), 1 - dur) # lower bound

#         a = np.round(self.sr * low, 0)
#         b = np.round(self.sr * dur, 0) + a

#         clip[:, a : b] = libr.effects.pitch_shift(clip[:, a : b], self.sr, n_steps = pitch) # may modify data matrix, not a huge deal


    def __len__(self):
        return self.data.shape[0]

    def augment_pitch(self, clip):
        """ Augment pitch and apply mu-law encoding to audio clip"""
        pitch = self.range * 2 * (np.random.random_sample() - 0.5) # how much to raise/lower by
        dur = (np.random.random_sample() / 4 + 0.25) * self.clip_length # duration of subsample between [0.25, .5]
        low = min(self.clip_length * np.random.random_sample(), self.clip_length - dur) # lower bound
        a = int(clip.shape[0] * low) 
        b = (int(clip.shape[0] * dur) + a)
        clip[a : b] = libr.effects.pitch_shift(clip[a : b], self.sr, n_steps = pitch) # may modify data matrix, not a huge deal
        
        clip = torch.Tensor(clip)
        return mu_law_encode(clip / utils.MAX_WAV_VALUE) # apply mu law encoding
