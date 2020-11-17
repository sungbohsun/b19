import os
import mir_eval
import pretty_midi as pm
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
import argparse
import warnings

#--------------------- CNN in test type------------------------
from utils.hparams import HParams
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from crf_model import CRF

use_cuda = torch.cuda.is_available()

class CNN(nn.Module):
    def __init__(self,config):
        super(CNN, self).__init__()

        self.timestep = config['timestep']
        self.context = 7
        self.pad = nn.ConstantPad1d(self.context, 0)
        self.probs_out = config['probs_out']
        self.num_chords = config['num_chords']

        self.drop_out = nn.Dropout2d(p=0.5)
        self.conv1 = self.cnn_layers(1, 32, kernel_size=(3,3), padding=1)
        self.conv2 = self.cnn_layers(32, 32, kernel_size=(3,3), padding=1)
        self.conv3 = self.cnn_layers(32, 32, kernel_size=(3,3), padding=1)
        self.conv4 = self.cnn_layers(32, 32, kernel_size=(3,3), padding=1)
        self.pool_max = nn.MaxPool2d(kernel_size=(2,1))
        self.conv5 = self.cnn_layers(32, 64, kernel_size=(3, 3), padding=0)
        self.conv6 = self.cnn_layers(64, 64, kernel_size=(3, 3), padding=0)
        self.conv7 = self.cnn_layers(64, 128, kernel_size=(12, 9), padding=0)
        self.conv_linear = nn.Conv2d(128, config['num_chords'], kernel_size=(1,1), padding=0)

    def cnn_layers(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        layers = []
        conv2d = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=padding)
        batch_norm = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        layers += [conv2d, batch_norm, relu]
        return nn.Sequential(*layers)

    #def forward(self, x, labels):
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pad(x)
        batch_size = x.size(0)
        for i in range(batch_size):
            for j in range(self.timestep):
                if i == 0 and j == 0:
                    inputs = x[i,:,j : j + self.context *2 + 1].unsqueeze(0)
                else:
                    tmp = x[i, :, j : j + self.context *2 + 1].unsqueeze(0)
                    inputs = torch.cat((inputs,tmp), dim=0)
        # inputs : [batchsize * timestep, feature_size, context]
        inputs = inputs.unsqueeze(1)
        conv = self.conv1(inputs)
        conv = self.conv2(conv)
        conv = self.conv3(conv)
        conv = self.conv4(conv)
        pooled = self.pool_max(conv)
        pooled = self.drop_out(pooled)
        conv = self.conv5(pooled)
        conv = self.conv6(conv)
        pooled = self.pool_max(conv)
        pooled = self.drop_out(pooled)
        conv = self.conv7(pooled)
        conv = self.drop_out(conv)
        conv = self.conv_linear(conv)
        avg_pool = nn.AvgPool2d(kernel_size=(conv.size(2), conv.size(3)))
        logits = avg_pool(conv).squeeze(2).squeeze(2)
        if self.probs_out is True:
            crf_input = logits.view(-1, self.timestep, self.num_chords)
            return crf_input
        log_probs = F.log_softmax(logits, -1)
        topk, indices = torch.topk(log_probs, 2)
        predictions = indices[:,0]
        second = indices[:,1]
        prediction = predictions.view(-1)
        second = second.view(-1)
#         loss = F.nll_loss(log_probs.view(-1, self.num_chords), labels.view(-1))
#         return prediction, loss, 0, second
        return prediction

class Crf(nn.Module):
    def __init__(self, num_chords, timestep):
        super(Crf, self).__init__()
        self.output_size = num_chords
        self.timestep = timestep
        self.Crf = CRF(self.output_size)

    #def forward(self, probs, labels):
    def forward(self, probs):
        prediction = self.Crf(probs)
        prediction = prediction.view(-1)
        #labels = labels.view(-1, self.timestep)
        #loss = self.Crf.loss(probs, labels)
        #return prediction, loss
        return prediction


class CRNN(nn.Module):
    def __init__(self,config):
        super(CRNN, self).__init__()

        self.feature_size = config['feature_size']
        self.timestep = config['timestep']
        self.probs_out = config['probs_out']
        self.num_chords = config['num_chords']
        self.hidden_size = 128

        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(5,5), padding=2)
        self.conv2 = nn.Conv2d(1, 36, kernel_size=(1,self.feature_size))
        self.gru = nn.GRU(input_size=36, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, self.num_chords)

    #def forward(self, x, labels):
    def forward(self, x):
        # x : [batchsize * timestep * feature_size]
        x = x.unsqueeze(1)
        x = self.batch_norm(x)
        conv = self.relu(self.conv1(x))
        conv = self.relu(self.conv2(conv))
        conv = conv.squeeze(3).permute(0,2,1)

        h0 = torch.zeros(4, conv.size(0), self.hidden_size).to(torch.device("cuda" if use_cuda else "cpu"))
        gru, h = self.gru(conv, h0)
        logits = self.fc(gru)
        if self.probs_out is True:
            # probs = F.softmax(logits, -1)
            return logits
        log_probs = F.log_softmax(logits, -1)
        topk, indices = torch.topk(log_probs, 2)
        predictions = indices[:,:,0]
        second = indices[:,:,1]
        prediction = predictions.view(-1)
        second = second.view(-1)
        #loss = F.nll_loss(log_probs.view(-1, self.num_chords), labels.view(-1))
        return prediction
#---------------------  ------------------------

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
    print('--- use cnn model ----')
elif args.model == 'crnn':
    model = CRNN(config=config.model).to(device)
elif args.model == 'btc':
    model = BTC_model(config=config.model).to(device)
else: raise NotImplementedError

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
            #np.save('feature2',feature[:, n_timestep * t:n_timestep * (t + 1), :].cpu())
            crf = Crf(170, timestep = config.model['timestep']).to(device)
            prediction = model(feature[:, n_timestep * t:n_timestep * (t + 1), :])
            #np.save('probs2',probs.cpu())
            #prediction = crf(probs)

            for i in range(n_timestep):
                if t == 0 and i == 0:
                    prev_chord = prediction[i].cpu().numpy().item()
                    continue
                if prediction[i].item() != prev_chord:
                    lines.append(
                        '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    start_time = time_unit * (n_timestep * t + i)
                    prev_chord = prediction[i].cpu().numpy().item()
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

