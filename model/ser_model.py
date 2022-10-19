import numpy as np
import torch
import torch.nn as nn

import librosa
import logging

from sklearn.preprocessing import StandardScaler


class TimeDistributed(nn.Module):

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        elif len(x.size()) == 3:
            x_reshape = x.contiguous().view(-1, x.size(2))
        elif len(x.size()) == 4:
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3))
        else:
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3), x.size(4))

        y = self.module(x_reshape)

        if len(x.size()) == 3:
            y = y.contiguous().view(x.size(0), -1, y.size(1))
        elif len(x.size()) == 4:
            y = y.contiguous().view(x.size(0), -1, y.size(1), y.size(2))
        else:
            y = y.contiguous().view(x.size(0), -1, y.size(1), y.size(2), y.size(3))
        return y


class HybridModel(nn.Module):

    def __init__(self, num_emotions):
        super().__init__()

        self.conv2d_block = nn.Sequential(
            # 1st Conv Block
            TimeDistributed(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            ),
            TimeDistributed(nn.BatchNorm2d(16)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
            TimeDistributed(nn.Dropout(0.3)),

            # 2nd Conv Block
            TimeDistributed(
                nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            ),
            TimeDistributed(nn.BatchNorm2d(32)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(0.3)),

            # 3rd Conv Block
            TimeDistributed(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            ),
            TimeDistributed(nn.BatchNorm2d(64)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(0.3)),
        )

        # LSTM Block
        hidden_size = 64
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.4)
        self.attention_linear = nn.Linear(2 * hidden_size, 1)

        # Linear Softmax Layer
        self.out_linear = nn.Linear(2 * hidden_size, num_emotions)

    def forward(self, x):
        conv_emb = self.conv2d_block(x)
        conv_emb = torch.flatten(conv_emb, start_dim=2)
        lstm_emb, (h, c) = self.lstm(conv_emb)
        lstm_emb = self.dropout_lstm(lstm_emb)

        batch_size, T, _ = lstm_emb.shape
        attention_weights = [None] * T

        for t in range(T):
            emb = lstm_emb[:, t, :]
            attention_weights[t] = self.attention_linear(emb)

        attention_weights_norm = nn.functional.softmax(torch.stack(attention_weights, -1), dim=-1)
        attention = torch.bmm(attention_weights_norm, lstm_emb)
        attention = torch.squeeze(attention, 1)
        output_logits = self.out_linear(attention)
        output_softmax = nn.functional.softmax(output_logits, dim=1)

        return output_logits, output_softmax, attention_weights_norm


def loss_fn(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)


class SER:
    EMOTIONS = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fear',
        7: 'disgust',
        0: 'surprised'
    }
    SAMPLE_RATE = 48000

    def __init__(self, mode='train', model_path=''):
        self.mode = mode

        self.model = HybridModel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_uc = self.model(len(self.EMOTIONS)).to(self.device)

        if self.mode == 'inference':
            if self.device == 'cuda':
                self.model_uc.load_state_dict(state_dict=torch.load(model_path))
            else:
                self.model_uc.load_state_dict(state_dict=torch.load(model_path, map_location=torch.device('cpu')))
            logging.info('Model : Loaded SER Model')

    @staticmethod
    def get_mel_spectrogram(audio, sample_rate):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_fft=1024, win_length=512,
            window='hamming', hop_length=256, n_mels=128, fmax=sample_rate / 2
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    @staticmethod
    def split_into_chunks(mel_spec, win_size, stride):
        t = mel_spec.shape[1]
        num_of_chunks = int(t / stride)

        chunks = []
        for i in range(num_of_chunks):
            chunk = mel_spec[:, i * stride:i * stride + win_size]
            if chunk.shape[1] == win_size:
                chunks.append(chunk)
        return np.stack(chunks, axis=0)

    @classmethod
    def load(cls, model_path):
        return cls(mode='inference', model_path=model_path)

    def predict(self, audio_file, sr=SAMPLE_RATE):
        audio, sample_rate = librosa.load(audio_file, sr=sr)

        mel_audio = self.get_mel_spectrogram(audio, sample_rate=sr)

        mel_audio_chunked = []
        chunks = self.split_into_chunks(mel_audio, win_size=128, stride=64)
        mel_audio_chunked.append(chunks)

        X_audio = np.stack(mel_audio_chunked, axis=0)
        X_audio = np.expand_dims(X_audio, 2)

        scaler = StandardScaler()

        b, t, c, h, w = X_audio.shape
        X_audio = np.reshape(X_audio, newshape=(b, -1))
        X_audio = scaler.fit_transform(X_audio)
        X_audio = np.reshape(X_audio, newshape=(b, t, c, h, w))

        X_audio_tensor = torch.tensor(X_audio, device=self.device).float()

        with torch.no_grad():
            output_logits, output_softmax, attention_weights_norm = self.model_uc(X_audio_tensor)
            predictions = torch.argmax(output_softmax, dim=1)

    def match_prediction(self, audio_file, match_result, sr=SAMPLE_RATE):
        duration = int(librosa.get_duration(filename=audio_file, sr=sr))
        audio, sample_rate = librosa.load(audio_file, duration=duration, sr=sr)

        signal = np.zeros((int(sr * duration, )))
        signal[:len(audio)] = audio

        mel_audio = self.get_mel_spectrogram(signal, sample_rate=sr)

        mel_audio_chunked = []
        chunks = self.split_into_chunks(mel_audio, win_size=128, stride=64)
        mel_audio_chunked.append(chunks)

        X_audio = np.stack(mel_audio_chunked, axis=0)
        X_audio = np.expand_dims(X_audio, 2)

        scaler = StandardScaler()

        b, t, c, h, w = X_audio.shape
        X_audio = np.reshape(X_audio, newshape=(b, -1))
        X_audio = scaler.fit_transform(X_audio)
        X_audio = np.reshape(X_audio, newshape=(b, t, c, h, w))

        X_audio_tensor = torch.tensor(X_audio, device=self.device).float()

        with torch.no_grad():
            output_logits, output_softmax, attention_weights_norm = self.model_uc(X_audio_tensor)
            predictions = torch.argmax(output_softmax, dim=1)

        if predictions in match_result:
            return True
        return False


