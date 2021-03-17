import torch
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

    
def read_dataset(path, class_num):
    with open(path, 'r') as rf:
        texts = []
        labels = []
        dataset = rf.read().split('\n')
        for data in dataset:
            if data.strip() == '':
                continue
            text, label = data.split('\t')
            text = text.strip()
            label = int(label.strip())
            label = 0 if label == class_num else label
            texts.append(text)
            labels.append(label)
    return texts, labels


def sampling_dataset(dataset, class_num, sample_num):
    sampled_data = []
    random.shuffle(dataset)
    stats = {label: 0 for label in range(class_num)}
    while True:
        if len(sampled_data) == class_num*sample_num:
            break
        data = dataset.pop(0)
        text, label = data
        if stats[label] >= sample_num:
            dataset.append(data)
        else:
            sampled_data.append(data)
    return sampled_data, dataset
