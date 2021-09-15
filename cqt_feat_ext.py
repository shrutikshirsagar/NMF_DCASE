import numpy as np
import os
import pandas as pd

audio_path = "/media/amrgaballah/Seagate Backup Plus Drive/Bosch/DCASE2017/audio/"
final_vec = np.empty((0,431))
for filename in os.listdir(audio_path):
    f = os.path.join(audio_path, filename)
    y, sr = librosa.load(f)
    C = np.abs(librosa.cqt(y, sr=sr, fmin=2,bins_per_octave = 24))
    print(C.shape)
    cqt = np.mean(C, axis=0)
    final_vec = np.vstack((final_vec, cqt))
print(final_vec.shape)
df = pd.Dataframe(final_vec)
df.write_csv(out_path + 'cqt_dcase2017.csv', index = None, header=None)
