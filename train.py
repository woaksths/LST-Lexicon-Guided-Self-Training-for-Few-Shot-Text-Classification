import argparse
import constant as config
import torch
from util.dataset import read_dataset, sampling_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import  DataLoader
from trainer import Trainer
from util.augment import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', required = True)
parser.add_argument('--test_path', required = True)
parser.add_argument('--save_path', required = True)
parser.add_argument('--do_augment', type=bool, default=False, required = False)
args = parser.parse_args()

# Read dataset
train_texts, train_labels = read_dataset(args.train_path, config.class_num)
test_texts, test_labels = read_dataset(args.test_path, config.class_num)

# Dataset sampling -> Since we are going to simulate semi-supervised learning, we will assume that we only know a little part of labeled data.
labeled_data, remained_data = sampling_dataset(list(zip(train_texts, train_labels)), class_num=config.class_num, sample_num=30)
dev_data, unlabeled_data = sampling_dataset(remained_data, class_num=config.class_num, sample_num=30)

print('labeled num {}, unlabeled num {}, valid num {}'.format(len(labeled_data), len(unlabeled_data), len(dev_data)))

# Tokenizing 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

labeled_texts = [data[0] for data in labeled_data]
labeled_labels = [data[1] for data in labeled_data]

if args.do_augment is True:
    augmented_texts, augmented_labels = back_translate(labeled_texts, labeled_labels)
    labeled_texts.extend(augmented_texts)
    labeled_labels.extend(augmented_labels)

labeled_encodings = tokenizer(labeled_texts, truncation=True, padding=True)
labeled_dataset = Dataset(labeled_encodings, labeled_labels)

dev_texts = [data[0] for data in dev_data]
dev_labels = [data[1] for data in dev_data]
dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)
dev_dataset = Dataset(dev_encodings, dev_labels)

test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = Dataset(test_encodings, test_labels)

# We keep the label of unlabeled data to track for accuracy of pseudo-labeling
unlabeled_texts = [data[0] for data in unlabeled_data]
unlabeled_labels = [data[1] for data in unlabeled_data]
unlabeled_encodings = tokenizer(unlabeled_texts, truncation=True, padding=True)
unlabeled_dataset = Dataset(unlabeled_encodings, unlabeled_labels)

# Build model 
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num)

# Criterion & optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5) #or AdamW

# Init Trainer
trainer = Trainer(config, model, loss_function, optimizer, args.save_path, dev_dataset, test_dataset)

# Initial training (supervised leraning)
trainer.initial_train(labeled_dataset)

# load checkpoint 
checkpoint_path = trainer.sup_path +'/checkpoint.pt'
checkpoint = torch.load(checkpoint_path)

del model, optimizer, trainer.model, trainer.optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

trainer.model = model
trainer.optimizer = optimizer

# eval supervised trained model 
trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)

# self-training
trainer.self_train(labeled_dataset, unlabeled_dataset)

# eval semi-supervised trained model 
checkpoint_path = trainer.ssl_path +'/checkpoint.pt'
checkpoint = torch.load(checkpoint_path)

del model, optimizer, trainer.model, trainer.optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

trainer.model = model
trainer.optimizer = optimizer

trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)
