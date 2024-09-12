import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import numpy as np
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    dataset = load_dataset("7wolf/gender-balanced-10k-voice-samples")
    return dataset

def extract_mfcc_features(batch):
    features = []
    for audio_dict in batch['audio']:
        waveform = audio_dict['array']
        sampling_rate = audio_dict['sampling_rate']
        
        # Extract MFCC features using librosa
        mfccs = librosa.feature.mfcc(y=waveform, sr=sampling_rate, n_mfcc=13)  # Adjust n_mfcc as needed
        # Transpose and pad/truncate features
        mfccs = mfccs.T
        
        features.append(mfccs)
    
    features_array = np.array([np.pad(f, ((0, 160 - f.shape[0]), (0, 0)), mode='constant') for f in features])  # Pad/truncate to a fixed length
    return {'features': features_array}

def apply_change():
    dataset = load_data()
    dataset = dataset.map(extract_mfcc_features, batched=True, batch_size=8)
    label_list = sorted(set(dataset['train']['label']))
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    
    return label_to_id, dataset

def label_to_int(batch, label_to_id):
    batch['label'] = [label_to_id.get(label, -1) for label in batch['label']]
    batch['label'] = [1 if l == 1 else 0 for l in batch['label']]
    return batch

def final_dataset():
    label_to_id, dataset = apply_change()
    dataset = dataset.map(lambda batch: label_to_int(batch, label_to_id), batched=True)
    return dataset

class AudioDataset(Dataset):
    def __init__(self, data):
        if 'features' in data.column_names:
            self.features = torch.tensor(np.vstack(data['features']), dtype=torch.float32)
        else:
            raise KeyError("'features' column is missing in the dataset.")
        
        if 'label' in data.column_names:
            self.labels = torch.tensor(data['label'], dtype=torch.long)
        else:
            raise KeyError("'label' column is missing in the dataset.")
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {'features': self.features[idx].to(device),
                'label': self.labels[idx].to(device)}

def get_loaders():
    dataset = final_dataset()
    train_data = AudioDataset(dataset['train'])
    test_data = AudioDataset(dataset['test'])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    input_dim = train_data.features.shape[1]  # This should be the length of the MFCC features
    return input_dim, train_loader, test_loader
