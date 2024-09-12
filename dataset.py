import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    # Load the dataset
    dataset = load_dataset("7wolf/gender-balanced-10k-voice-samples")
    
    # Remove the 'id' column and select a subset
    train_subset = dataset['train'].select(range(5)).remove_columns(['id'])
    test_subset = dataset['test'].select(range(2)).remove_columns(['id'])
    
    subset_dataset = DatasetDict({
        'train': train_subset,
        'test': test_subset
    })
    return subset_dataset

def extract_wav2vec_features(batch):
    # Initialize processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

    features = []
    for audio_dict in batch['audio']:
        waveform = audio_dict['array']
        sampling_rate = audio_dict['sampling_rate']
        
        # Process audio and get features
        inputs = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True, truncation=True, max_length=160000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        feature = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        features.append(feature)
    
    # Return features as a list
    return {'features': features}

def apply_change():
    dataset = load_data()
    
    # Apply feature extraction
    dataset = dataset.map(extract_wav2vec_features, batched=True, batch_size=8)
    
    # Verify columns after feature extraction
    print("Columns after feature extraction:", dataset['train'].column_names)
    
    label_list = sorted(set(dataset['train']['label']))
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    
    return label_to_id, dataset

def label_to_int(batch, label_to_id):
    unknown_label = -1  
    batch['label'] = [label_to_id.get(label, unknown_label) for label in batch['label']]
    return batch

def final_dataset():
    label_to_id, dataset = apply_change()
    
    # Map label to int
    dataset = dataset.map(lambda batch: label_to_int(batch, label_to_id), batched=True)
    
    # Verify final columns in dataset
    print("Final columns in dataset:", dataset['train'].column_names)
    
    return dataset

class AudioDataset(Dataset):
    def __init__(self, data):
        # Debugging: Print data structure
        print("Data sample:", data)
        
        # Ensure 'features' column exists
        if 'features' in data:
            self.features = torch.tensor(np.vstack(data['features']), dtype=torch.float32)
        else:
            raise KeyError("'features' column is missing in the dataset.")
        
        # Ensure 'label' column exists
        if 'label' in data:
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
    
    # Debugging: Print dataset structure before creating loaders
    print("Dataset sample:", dataset['train'][0])
    
    train_data = AudioDataset(dataset['train'])
    test_data = AudioDataset(dataset['test'])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    input_dim = train_data.features.shape[1]  # Get the feature dimension
    return input_dim, train_loader, test_loader
