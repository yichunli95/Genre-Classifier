import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

'''
    function: extract_features
    input: path to mp3 files
    output: csv file containing features extracted
    
    This function reads the content in a directory and for each mp3 file detected
    reads the file and extracts relevant features using librosa library for audio
    signal processing
'''
def extract_feature(path):
    id = 1  # Song ID
    feature_set = pd.DataFrame()  # Feature Matrix
    
    # Individual Feature Vectors
    #songname_vector = pd.Series()
    #tempo_vector = pd.Series()
    #total_beats = pd.Series()
    #average_beats = pd.Series()
    chroma_stft_mean = pd.Series()
    #chroma_stft_std = pd.Series()
    #chroma_stft_var = pd.Series()
    #chroma_cq_mean = pd.Series()
    #chroma_cq_std = pd.Series()
    #chroma_cq_var = pd.Series()
    #chroma_cens_mean = pd.Series()
    #chroma_cens_std = pd.Series()
    #chroma_cens_var = pd.Series()
    mel_mean = pd.Series()
    #mel_std = pd.Series()
    #mel_var = pd.Series()
    mfcc_mean = pd.Series()
    #mfcc_std = pd.Series()
    #mfcc_var = pd.Series()
    #mfcc_delta_mean = pd.Series()
    #mfcc_delta_std = pd.Series()
    #mfcc_delta_var = pd.Series()
    rmse_mean = pd.Series()
    #rmse_std = pd.Series()
    #rmse_var = pd.Series()
    cent_mean = pd.Series()
    #cent_std = pd.Series()
    #cent_var = pd.Series()
    #spec_bw_mean = pd.Series()
    #spec_bw_std = pd.Series()
    #spec_bw_var = pd.Series()
    contrast_mean = pd.Series()
    #contrast_std = pd.Series()
    #contrast_var = pd.Series()
    rolloff_mean = pd.Series()
    #rolloff_std = pd.Series()
    #rolloff_var = pd.Series()
    #poly_mean = pd.Series()
    #poly_std = pd.Series()
    #poly_var = pd.Series()
    #tonnetz_mean = pd.Series()
    #tonnetz_std = pd.Series()
    #tonnetz_var = pd.Series()
    #zcr_mean = pd.Series()
    zcr_std = pd.Series()
    #zcr_var = pd.Series()
    #harm_mean = pd.Series()
    harm_std = pd.Series()
    #harm_var = pd.Series()
    #perc_mean = pd.Series()
    perc_std = pd.Series()
    #perc_var = pd.Series()
    frame_mean = pd.Series()
    #frame_std = pd.Series()
    #frame_var = pd.Series()
    
    
    # Traversing over each file in path
    file_data = [f for f in listdir(path) if isfile (join(path, f))]
    for line in file_data:
        if ( line[-1:] == '\n' ):
            line = line[:-1]

        # Reading Song
        songname = path + line
        y, sr = librosa.load(songname, duration=60)
        S = np.abs(librosa.stft(y))
        
        # Extracting Features
        #tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        #chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        #chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        #spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        #poly_features = librosa.feature.poly_features(S=S, sr=sr)
        #tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        #mfcc_delta = librosa.feature.delta(mfcc)
    
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
        
        # Transforming Features
        #songname_vector.set_value(id, line)  # song name
        #tempo_vector.set_value(id, tempo)  # tempo
        #total_beats.set_value(id, sum(beats))  # beats
        #average_beats.set_value(id, np.average(beats))
        chroma_stft_mean.set_value(id, (np.mean(chroma_stft) - 0.181137412) / 0.482547326)  # chroma stft
        #chroma_stft_std.set_value(id, np.std(chroma_stft))
        #chroma_stft_var.set_value(id, np.var(chroma_stft))
        #chroma_cq_mean.set_value(id, np.mean(chroma_cq))  # chroma cq
        #chroma_cq_std.set_value(id, np.std(chroma_cq))
        #chroma_cq_var.set_value(id, np.var(chroma_cq))
        #chroma_cens_mean.set_value(id, np.mean(chroma_cens))  # chroma cens
        #chroma_cens_std.set_value(id, np.std(chroma_cens))
        #chroma_cens_var.set_value(id, np.var(chroma_cens))
        mel_mean.set_value(id, (np.mean(melspectrogram) - 0.013551708) / 22.97888649)  # melspectrogram
        #mel_std.set_value(id, np.std(melspectrogram))
        #mel_var.set_value(id, np.var(melspectrogram))
        mfcc_mean.set_value(id, (np.mean(mfcc) - (-15.60405998)) / 23.9358552)  # mfcc
        #mfcc_std.set_value(id, np.std(mfcc))
        #mfcc_var.set_value(id, np.var(mfcc))
       # mfcc_delta_mean.set_value(id, np.mean(mfcc_delta))  # mfcc delta
        #mfcc_delta_std.set_value(id, np.std(mfcc_delta))
        #mfcc_delta_var.set_value(id, np.var(mfcc_delta))
        rmse_mean.set_value(id, (np.mean(rmse) - 0.146055445) / 9.06019856)  # rmse
        #rmse_std.set_value(id, np.std(rmse))
        #rmse_var.set_value(id, np.var(rmse))
        cent_mean.set_value(id, (np.mean(cent) - 570.0403441) / 3865.204066)  # cent
       # cent_std.set_value(id, np.std(cent))
        #cent_var.set_value(id, np.var(cent))
        #spec_bw_mean.set_value(id, np.mean(spec_bw))  # spectral bandwidth
        #spec_bw_std.set_value(id, np.std(spec_bw))
        #spec_bw_var.set_value(id, np.var(spec_bw))
        contrast_mean.set_value(id, (np.mean(contrast) - 13.51248905) / 13.20251862)  # contrast
        #contrast_std.set_value(id, np.std(contrast))
        #contrast_var.set_value(id, np.var(contrast))
        rolloff_mean.set_value(id, (np.mean(rolloff) - 927.6513898) / 7329.474344)  # rolloff
        #rolloff_std.set_value(id, np.std(rolloff))
        #rolloff_var.set_value(id, np.var(rolloff))
       # poly_mean.set_value(id, np.mean(poly_features))  # poly features
       # poly_std.set_value(id, np.std(poly_features))
        #poly_var.set_value(id, np.var(poly_features))
       # tonnetz_mean.set_value(id, np.mean(tonnetz))  # tonnetz
       # tonnetz_std.set_value(id, np.std(tonnetz))
        #tonnetz_var.set_value(id, np.var(tonnetz))
       # zcr_mean.set_value(id, np.mean(zcr))  # zero crossing rate
        zcr_std.set_value(id, (np.std(zcr) - 0.010820747) / 0.13083421)
        #zcr_var.set_value(id, np.var(zcr))
      #  harm_mean.set_value(id, np.mean(harmonic))  # harmonic
        harm_std.set_value(id, (np.std(harmonic) - 0.00703311) / 0.24275273)
        #harm_var.set_value(id, np.var(harmonic))
       # perc_mean.set_value(id, np.mean(percussive))  # percussive
        perc_std.set_value(id, (np.std(percussive) - 0.001101192) / 0.21671978)
        #perc_var.set_value(id, np.var(percussive))
        frame_mean.set_value(id, (np.mean(frames_to_time) - 1.214403628) / 8.152526077)  # frames
       # frame_std.set_value(id, np.std(frames_to_time))
        #frame_var.set_value(id, np.var(frames_to_time))
        
        print(songname)
        id = id+1
    
    # Concatenating Features into one csv and json format
    #feature_set['song_name'] = songname_vector  # song name
    #feature_set['tempo'] = tempo_vector  # tempo 
    #feature_set['total_beats'] = total_beats  # beats
    #feature_set['average_beats'] = average_beats
    feature_set['chroma_stft_mean'] = chroma_stft_mean  # chroma stft
    #feature_set['chroma_stft_std'] = chroma_stft_std
    #feature_set['chroma_stft_var'] = chroma_stft_var
   # feature_set['chroma_cq_mean'] = chroma_cq_mean  # chroma cq
    #feature_set['chroma_cq_std'] = chroma_cq_std
    #feature_set['chroma_cq_var'] = chroma_cq_var
   # feature_set['chroma_cens_mean'] = chroma_cens_mean  # chroma cens
   # feature_set['chroma_cens_std'] = chroma_cens_std
    #feature_set['chroma_cens_var'] = chroma_cens_var
    feature_set['melspectrogram_mean'] = mel_mean  # melspectrogram
   # feature_set['melspectrogram_std'] = mel_std
    #feature_set['melspectrogram_var'] = mel_var
    feature_set['mfcc_mean'] = mfcc_mean  # mfcc
   # feature_set['mfcc_std'] = mfcc_std
    #feature_set['mfcc_var'] = mfcc_var
    #feature_set['mfcc_delta_mean'] = mfcc_delta_mean  # mfcc delta
    #feature_set['mfcc_delta_std'] = mfcc_delta_std
   # f#eature_set['mfcc_delta_var'] = mfcc_delta_var
    feature_set['rmse_mean'] = rmse_mean  # rmse
    #feature_set['rmse_std'] = rmse_std
    #feature_set['rmse_var'] = rmse_var
    feature_set['cent_mean'] = cent_mean  # cent
   # feature_set['cent_std'] = cent_std
    #feature_set['cent_var'] = cent_var
   # feature_set['spec_bw_mean'] = spec_bw_mean  # spectral bandwidth
   # feature_set['spec_bw_std'] = spec_bw_std
   # f#eature_set['spec_bw_var'] = spec_bw_var
    feature_set['contrast_mean'] = contrast_mean  # contrast
   # feature_set['contrast_std'] = contrast_std
    #feature_set['contrast_var'] = contrast_var
    feature_set['rolloff_mean'] = rolloff_mean  # rolloff
   # feature_set['rolloff_std'] = rolloff_std
   # f#eature_set['rolloff_var'] = rolloff_var
   # feature_set['poly_mean'] = poly_mean  # poly features
   # feature_set['poly_std'] = poly_std
    #feature_set['poly_var'] = poly_var
   # feature_set['tonnetz_mean'] = tonnetz_mean  # tonnetz
   # feature_set['tonnetz_std'] = tonnetz_std
    #feature_set['tonnetz_var'] = tonnetz_var
   # feature_set['zcr_mean'] = zcr_mean  # zero crossing rate
    feature_set['zcr_std'] = zcr_std
    #feature_set['zcr_var'] = zcr_var
   # feature_set['harm_mean'] = harm_mean  # harmonic
    feature_set['harm_std'] = harm_std
    #feature_set['harm_var'] = harm_var
   # feature_set['perc_mean'] = perc_mean  # percussive
    feature_set['perc_std'] = perc_std
    #feature_set['perc_var'] = perc_var
    feature_set['frame_mean'] = frame_mean  # frames
   # feature_set['frame_std'] = frame_std
    #feature_set['frame_var'] = frame_var

    # Converting Dataframe into CSV Excel and JSON file
    feature_set.to_csv('Emotion_features.csv')
    feature_set.to_json('Emotion_features.json')
    
# Extracting Feature Function Call
extract_feature('Dataset/')