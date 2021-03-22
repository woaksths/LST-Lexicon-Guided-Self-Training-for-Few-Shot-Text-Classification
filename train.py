import argparse
import constant as config
import torch
from util.dataset import read_dataset, sampling_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import  DataLoader
from trainer import Trainer
from util.augment import *
import copy

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

train_texts = copy.deepcopy(labeled_texts)
train_labels = copy.deepcopy(labeled_labels)

if args.do_augment is True:
    augmented_texts, augmented_labels = back_translate(labeled_texts, labeled_labels)
    train_texts.extend(augmented_texts)
    train_labels.extend(augmented_labels)
    
    augmented_texts, augmented_labels = word_replacement(labeled_texts, labeled_labels)
    train_texts.extend(augmented_texts)
    train_labels.extend(augmented_labels)
    
    print(len(train_texts), len(train_labels))
    
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_dataset = Dataset(train_encodings, train_labels)

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

labeled_encodings = tokenizer(labeled_texts, truncation=True, padding=True)
labeled_dataset = Dataset(labeled_encodings, labeled_labels)

# Build model 
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num)

# Criterion & optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5) #or AdamW

# Init Trainer
trainer = Trainer(config, model, loss_function, optimizer, args.save_path, dev_dataset, test_dataset)

# Initial training (supervised leraning)
trainer.initial_train(train_dataset)

# load sup checkpoint 
del model, optimizer, trainer.model, trainer.optimizer
model = BertForSequenceClassification.from_pretrained(trainer.sup_path).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

trainer.model = model
trainer.optimizer = optimizer

# eval supervised trained model 
trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)

# self-training -> guide_type = ['predefined_lexicon', 'generated_lexicon', 'naive_bayes', 'advanced_nb']
trainer.self_train(labeled_dataset, unlabeled_dataset, guide_type= 'predefined_lexicon')

# load ssl checkpoint
del model, optimizer, trainer.model, trainer.optimizer
model = BertForSequenceClassification.from_pretrained(trainer.ssl_path).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

trainer.model = model
trainer.optimizer = optimizer

# eval semi-supervised trained model 
trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)
