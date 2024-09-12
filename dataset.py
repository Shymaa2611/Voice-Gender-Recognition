import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load data with only 'audio' and 'label' columns
def load_data():
    dataset = load_dataset("7wolf/gender-balanced-10k-voice-samples")
    
    # Select only 'audio' and 'label' columns
    train_subset = dataset['train'].select(range(5)).remove_columns([col for col in dataset['train'].column_names if col not in ['audio', 'label']])
    test_subset = dataset['test'].select(range(2)).remove_columns([col for col in dataset['test'].column_names if col not in ['audio', 'label']])
    
    subset_dataset = DatasetDict({
        'train': train_subset,
        'test': test_subset
    })
    return subset_dataset

# Step 2: Extract Wav2Vec2 features
def extract_wav2vec_features(batch):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

    features = []
    for audio_dict in batch['audio']:
        waveform = audio_dict['array']
        sampling_rate = audio_dict['sampling_rate']
        
        inputs = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True, truncation=True, max_length=160000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        feature = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        features.append(feature)

    # Return the extracted features as a new column in the dataset
    return {'features': features}

# Step 3: Apply changes and ensure dataset contains 'audio', 'label', 'features'
def apply_change():
    dataset = load_data()

    # Apply feature extraction and keep only the 'audio', 'label', and 'features' columns
    dataset = dataset.map(extract_wav2vec_features, batched=True, batch_size=8)
    
    # Debugging: Check if features are added to the dataset
    print("Columns after feature extraction:", dataset['train'].column_names)
    
    label_list = sorted(set(dataset['train']['label']))
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    
    return label_to_id, dataset

# Step 4: Label conversion
def label_to_int(batch, label_to_id):
    unknown_label = -1  
    batch['label'] = [label_to_id.get(label, unknown_label) for label in batch['label']]
    return batch

# Step 5: Final dataset with features and integer labels
def final_dataset():
    label_to_id, dataset = apply_change()
    
    # Map labels to integers
    dataset = dataset.map(lambda batch: label_to_int(batch, label_to_id), batched=True)
    
    # Debugging: Verify final columns in dataset
    print("Final columns in dataset:", dataset['train'].column_names)
    
    return dataset

# Step 6: Dataset class
class AudioDataset(Dataset):
    def __init__(self, data):
        # Check if the 'features' column exists and stack the features into a tensor
        if 'features' in data:
            self.features = torch.tensor(np.vstack(data['features']), dtype=torch.float32)
        else:
            raise KeyError("'features' column is missing in the dataset.")
        self.labels = torch.tensor(data['label'], dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {'features': self.features[idx].to(device),
                'label': self.labels[idx].to(device)}

# Step 7: DataLoader setup
def get_loaders():
    dataset = final_dataset()
    train_data = AudioDataset(dataset['train'])
    test_data = AudioDataset(dataset['test'])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    input_dim = train_data.features.shape[1]  # Get the feature dimension
    return input_dim, train_loader, test_loader
