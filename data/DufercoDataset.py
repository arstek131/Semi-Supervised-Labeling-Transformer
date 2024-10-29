import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class DufercoDataset(Dataset):
    def __init__(self, data_config_path, split, transform=None):
 
        self.transform = transform
        self.label_mapping = {"aligned": 1, "not_aligned": 0}
        
        with open(data_config_path, 'r') as f:
            data_config = json.load(f)
        self.dataset = data_config.get(split, {})
        self.image_paths = list(self.dataset.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.label_mapping.get(self.dataset[img_path], 0)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_sample_weights(self):
        class_counts = [0, 0]
        
        for img_path in self.image_paths:
            label = self.label_mapping.get(self.dataset[img_path], 0)
            class_counts[label] += 1

        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        return [class_weights[self.label_mapping[self.dataset[img_path]]] for img_path in self.image_paths]
