import os
import mir_eval
import pretty_midi as pm
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
import argparse
import warnings
from baseline_models import CNN, CRNN

warnings.filterwarnings('ignore')
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--voca', default=True, type=lambda x: (str(x).lower() == 'true'))
#parser.add_argument('--pre', type=bool,default=False)
parser.add_argument('--model', default='btc')
parser.add_argument('--index', default='c')
parser.add_argument('--audio_dir', type=str, default='./test')
parser.add_argument('--save_dir', type=str, default='./test')
parser.add_argument('--model_file', type=str, default='model_file')
parser.add_argument('--normal_file', type=str, default='normal_file')
args = parser.parse_args()

config = HParams.load("config/run_config_idx{}.yaml".format(args.index))



config.feature['large_voca'] = True
config.model['num_chords'] = 170
if args.model == 'cnn':
    model = CNN(config=config.model).to(device)
elif args.model == 'crnn':
    model = CRNN(config=config.model).to(device)
elif args.model == 'btc':
    model = BTC_model(config=config.model).to(device)
else: raise NotImplementedError
#print('==========',args.pre)
#if args.pre :
    
#     config.model['num_chords'] = 170
#     model_file = './data/assets/model/'+args.model_file
#     idx_to_chord = idx2voca_chord()
#     checkpoint = torch.load(model_file)
#     mean = checkpoint['mean']
#     std = checkpoint['std']
#     model.load_state_dict(checkpoint['model'])
#     logger.info("restore model")

#else:    
model_file = './data/assets/model/'+args.model_file
normal_file = './data/result/'+args.normal_file

idx_to_chord = idx2voca_chord()    
checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint['model'])

checkpoint = torch.load(normal_file)
mean = checkpoint['mean']
std = checkpoint['std']
logger.info("restore model")

# Audio files with format of wav and mp3
audio_paths = get_audio_paths('./test/mp3/'+args.audio_dir)

# Chord recognition and save lab file
for i, audio_path in enumerate(audio_paths):
    logger.info("======== %d of %d in progress ========" % (i + 1, len(audio_paths)))
    # Load mp3
    feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
    logger.info("audio file loaded and feature computation success : %s" % audio_path)

    # Majmin type chord recognition
    feature = feature.T
    feature = (feature - mean) / std
    time_unit = feature_per_second
    n_timestep = config.model['timestep']

    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    start_time = 0.0
    lines = []
    with torch.no_grad():
        model.eval()
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            for i in range(n_timestep):
                if t == 0 and i == 0:
                    prev_chord = prediction[i].item()
                    continue
                if prediction[i].item() != prev_chord:
                    lines.append(
                        '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    start_time = time_unit * (n_timestep * t + i)
                    prev_chord = prediction[i].item()
                if t == num_instance - 1 and i + num_pad == n_timestep:
                    if start_time != time_unit * (n_timestep * t + i):
                        lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    break

    # lab file write
    save_dir = './test/result/'+args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav', '') + '.lab')
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)

    logger.info("label file saved : %s" % save_path)

    # lab file to midi file
    

    starts, ends, pitchs = list(), list(), list()

    intervals, chords = mir_eval.io.load_labeled_intervals(save_path)
    for p in range(12):
        for i, (interval, chord) in enumerate(zip(intervals, chords)):
            root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
            tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]
            if i == 0:
                start_time = interval[0]
                label = tmp_label
                continue
            if tmp_label != label:
                if label == 1.0:
                    starts.append(start_time), ends.append(interval[0]), pitchs.append(p + 48)
                start_time = interval[0]
                label = tmp_label
            if i == (len(intervals) - 1): 
                if label == 1.0:
                    starts.append(start_time), ends.append(interval[1]), pitchs.append(p + 48)

    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(program=0)

    for start, end, pitch in zip(starts, ends, pitchs):
        pm_note = pm.Note(velocity=120, pitch=pitch, start=start, end=end)
        instrument.notes.append(pm_note)

    midi.instruments.append(instrument)
    midi.write(save_path.replace('.lab', '.midi'))    

