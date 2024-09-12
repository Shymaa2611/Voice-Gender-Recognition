import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    dataset = load_dataset("7wolf/gender-balanced-10k-voice-samples")
    return dataset

def extract_wav2vec_features(batch):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

    audio_dicts = batch['audio']
    features = []
    for audio_dict in audio_dicts:
        if isinstance(audio_dict, dict):
            waveform = audio_dict.get('array', None)
            sampling_rate = audio_dict.get('sampling_rate', None)
            if waveform is not None and sampling_rate is not None:
                inputs = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True, truncation=True, max_length=160000)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                feature = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                features.append(feature)
            else:
                print("Missing waveform or sampling rate in audio_dict")
        else:
            print("Unexpected format for audio_dict:", type(audio_dict))
    return {'features': np.array(features)}

def apply_change():
    dataset = load_data()
    dataset = dataset.map(extract_wav2vec_features, batched=True)
    label_list = sorted(set(dataset['train']['label']))
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    return label_to_id

def label_to_int(batch):
    label_to_id = apply_change()
    unknown_label = -1  
    if isinstance(batch['label'], list):
        batch['label'] = [label_to_id.get(label, unknown_label) for label in batch['label']]
    else:
        batch['label'] = label_to_id.get(batch['label'], unknown_label)
    
    return batch

def final_dataset():
    dataset = load_data()
    dataset = dataset.map(label_to_int)
    return dataset

class AudioDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(data['features'], dtype=torch.float32)
        self.labels = torch.tensor(data['label'], dtype=torch.long)
        
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
    dim = 54614  
    return dim, train_loader, test_loader