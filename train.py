import argparse
import constant as config
import torch
from util.dataset import read_dataset, sampling_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from model import BERT_ATTN
from torch.utils.data import  DataLoader
from trainer import Trainer
# from util.augment import *
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', required = True)
parser.add_argument('--test_path', required = True)
parser.add_argument('--save_path', required = True)
parser.add_argument('--model_type', required=True, help='type among [baseline, bert_attn]')
parser.add_argument('--do_augment', type=bool, default=False, required = False)
args = parser.parse_args()

# Read dataset
train_texts, train_labels = read_dataset(args.train_path, config.class_num)
test_texts, test_labels = read_dataset(args.test_path, config.class_num)

# dataset sampling -> Since we are going to simulate semi-supervised learning, we will assume that we only know a little part of labeled data.
labeled_data, remained_data = sampling_dataset(list(zip(train_texts, train_labels)), class_num=config.class_num, sample_num=30)
dev_data, unlabeled_data = sampling_dataset(remained_data, class_num=config.class_num, sample_num=30)

print('labeled num {}, valid num {},  unlabeled num {}'.format(len(labeled_data), len(dev_data), len(unlabeled_data)))

# Tokenizing 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_texts = [data[0] for data in labeled_data]
train_labels = [data[1] for data in labeled_data]
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

# Build model 
if args.model_type.lower() == 'baseline':
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num)
else:
    model = BERT_ATTN(num_labels=config.class_num) 
    
# Criterion & optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5) #or AdamW

# Init Trainer
trainer = Trainer(config, model, loss_function, optimizer, 
                  args.save_path, dev_dataset, test_dataset,
                  args.model_type, args.do_augment)

# Initial training (supervised leraning)
trainer.initial_train(train_dataset)

# load sup checkpoint 
del model, optimizer, trainer.model, trainer.optimizer
if args.model_type.lower() == 'baseline':
    model = BertForSequenceClassification.from_pretrained(trainer.sup_path).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
else:
    model = BERT_ATTN(num_labels=config.class_num).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    checkpoint_path = trainer.sup_path +'/checkpoint.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

trainer.model = model
trainer.optimizer = optimizer

# eval supervised trained model 
trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)

# self-training -> guide_type = ['predefined_lexicon_pl', 'lexicon_pl', 'weighted_lexicon_pl']
trainer.self_train(train_dataset, list(zip(unlabeled_texts, unlabeled_labels)), guide_type= 'lexicon_pl', confidence_threshold=0.9)

# load ssl checkpoint
del model, optimizer, trainer.model, trainer.optimizer
if args.model_type.lower() == 'baseline':
    model = BertForSequenceClassification.from_pretrained(trainer.ssl_path).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
else:
    model = BERT_ATTN(num_labels=config.class_num).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    checkpoint_path = trainer.ssl_path +'/checkpoint.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

trainer.model = model
trainer.optimizer = optimizer

# eval semi-supervised trained model 
trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)
