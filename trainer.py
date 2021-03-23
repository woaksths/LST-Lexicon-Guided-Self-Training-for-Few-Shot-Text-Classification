import os
import torch
from evaluator import Evaluator
from torch.utils.data import  DataLoader
from util.early_stopping import EarlyStopping
from transformers import BertTokenizer, BertForSequenceClassification
from util.dataset import Dataset
from weak_supervision import guide_pseudo_labeling
import random

class Trainer(object):
    def __init__(self, config, model, criterion, optimizer, save_path, dev_dataset, test_dataset):
        self.config = config
        self.loss = criterion
        self.evaluator = Evaluator(loss=self.loss, batch_size=self.config.test_batch_size)
        self.optimizer = optimizer
        self.device = self.config.device
        self.model = model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.train_loader = None
        
        self.valid_loader = DataLoader(dev_dataset, **self.config.valid_params)
        self.test_loader = DataLoader(test_dataset, **self.config.test_params)
        
        self.early_stopping = None
        
        self.save_path = save_path
        self.sup_path = self.save_path +'/sup'
        self.ssl_path = self.save_path +'/ssl'
        
        if not os.path.isabs(self.sup_path):
            self.sup_path = os.path.join(os.getcwd(), self.sup_path)
        if not os.path.exists(self.sup_path):
            os.makedirs(self.sup_path)
        
        if not os.path.isabs(self.ssl_path):
            self.ssl_path = os.path.join(os.getcwd(), self.ssl_path)
        if not os.path.exists(self.ssl_path):
            os.makedirs(self.ssl_path)

        
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
        
    def train_epoch(self, epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        self.model.train()
        print('train_epoch', epoch)
        
        for _, batch in enumerate(self.train_loader):
            ids = batch['input_ids'].to(self.device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
            targets = batch['labels'].to(self.device, dtype=torch.long)
            
            outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
            loss, logits = outputs[0], outputs[1]
            
            tr_loss += loss.item()
            scores = torch.softmax(logits, dim=-1)
            big_val, big_idx = torch.max(scores.data, dim=-1)
            n_correct += self.calculate_accu(big_idx,targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if _ % 1000 == 0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Training Loss per 1000 steps: {loss_step}")
                print(f"Training Accuracy per 1000 steps: {accu_step}")
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")
        
        
    def initial_train(self, label_dataset):
        print('initial train module')
        self.train_loader = DataLoader(label_dataset, **self.config.train_params)
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        best_dev_acc = -1
        
        for epoch in range(self.config.epochs):
            self.train_epoch(epoch)
            dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.valid_loader)
            self.early_stopping(dev_loss)
            
            if best_dev_acc <= dev_acc:
                best_dev_acc = dev_acc
                self.model.save_pretrained(self.sup_path)

            if epoch % 1 == 0:
                test_loss, test_acc = self.evaluator.evaluate(self.model, self.test_loader, is_test=True)
            
            if self.early_stopping.early_stop:
                print("Eearly Stopping!")
                break

                
    def self_train(self, labeled_dataset, unlabeled_dataset, guide_type=None, confidence_threshold=0.9):
        best_accuracy = -1
        min_dev_loss = 987654321
        print(len(unlabeled_dataset))
        print(type(unlabeled_dataset))
        
        for outer_epoch in range(self.config.epochs):
            sampled_num = len(unlabeled_dataset) // 5
            random.shuffle(unlabeled_dataset)            
            sampled_unlabeled = unlabeled_dataset[:sampled_num]
            
            sampled_text = [data[0] for data in sampled_unlabeled]
            sampled_labels = [data[1] for data in sampled_unlabeled]
            sampled_encodings = self.tokenizer(sampled_text, truncation=True, padding=True)
            sampled_unlabeled_dataset = Dataset(sampled_encodings, sampled_labels)
            
            print('outer_epoch {} sampled unlabeled dataset {}'.format(outer_epoch, len(sampled_unlabeled_dataset)))
            
            # pseudo-labeling
            new_dataset = self.pseudo_labeling(sampled_unlabeled_dataset, confidence_threshold, guide_type)
            
            # add pseudo-label into labeled data
            combined_dataset, new_dataset = self.add_dataset(labeled_dataset, new_dataset)
            
            # remove pseudo-label from unlabeled data
            # unlabeled_dataset = self.remove_dataset(unlabeled_dataset, new_dataset)
            
            self.train_loader = DataLoader(combined_dataset, **self.config.train_params)
            self.early_stopping = EarlyStopping(patience=5, verbose=True)
            
            # re-initialize the student model from scratch 
            
            
            # retrain model with labeled data + pseudo-labeled data
            for inner_epoch in range(self.config.epochs):
                print('outer_epoch {} inner_epoch {}'.format(outer_epoch, inner_epoch))
                self.train_epoch(inner_epoch)
                dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.valid_loader)
                self.early_stopping(dev_loss)
                
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    self.model.save_pretrained(self.sup_path)
                
                if inner_epoch % 2 == 0:
                    test_loss, test_acc = self.evaluator.evaluate(self.model, self.test_loader, is_test=True)
                    if best_accuracy < test_acc:
                        best_accuracy = test_acc
                        
                if self.early_stopping.early_stop:
                    print("Early Stopping!")
                    break
                    
        print('Best accuracy {}'.format(best_accuracy))
    
    
    def pseudo_labeling(self, unlabeled_dataset, confidence_threshold, guide_type=None):
        unlabeled_loader = DataLoader(unlabeled_dataset, **self.config.unlabeled_params)
        self.model.eval()
        new_dataset = {label:[] for label in range(self.config.class_num)}
        
        with torch.no_grad():
            for _, batch in enumerate(unlabeled_loader):
                ids = batch['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                targets = batch['labels'].to(self.device, dtype=torch.long)
                
                outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
                loss, logits = outputs[0], outputs[1]
                confidences = torch.softmax(logits, dim=-1)
                big_val, big_idx = torch.max(confidences.data, dim=-1)

                for text_id, label, conf_val, target in zip(ids, big_idx, big_val, targets):
                    pred_label, conf_val, target = label.item(), conf_val.item(), target.item()
                    if conf_val >= confidence_threshold:
                        decoded_text = self.tokenizer.decode(text_id)
                        decoded_text = decoded_text.replace("[CLS]", "").replace("[SEP]","").replace("[PAD]","").strip()
                        new_dataset[pred_label].append((text_id, decoded_text, pred_label, target, conf_val))
                
        if guide_type == 'predefined_lexicon':
            new_dataset = guide_pseudo_labeling(new_dataset, guide_type)
        elif guide_type =='generated_lexicon':
            pass
        elif guide_type == 'naive_bayes':
            pass
        elif guide_type == 'advanced_nb':
            pass
        
        # make new_dataset being balanced 
        num_of_min_dataset = 987654321
        
        for label, dataset in new_dataset.items():
            if num_of_min_dataset > len(dataset):
                num_of_min_dataset = len(dataset)
        
        # sampling top N 
        top_N = 1000
        num_of_min_dataset = min(top_N, num_of_min_dataset)
        print('num_of_min_dataset', num_of_min_dataset)

        total, correct = 0, 0
        balanced_dataset = []
        
        for label in new_dataset.keys():
            # sort by confidence
            new_dataset[label].sort(key=lambda x:x[4], reverse=True)
            balanced_dataset.extend(new_dataset[label][:num_of_min_dataset])        
                
        for data in balanced_dataset:
            text_id, decoded_text, pred_label, target, confidence = data[0], data[1], data[2], data[3], data[4]
            if pred_label == target:
                correct+=1
            total+=1
        
        print('#'*100)
        print(' pseduo-label {}/{}'.format(correct, total))
        return balanced_dataset

    
    def add_dataset(self, labeled_dataset, new_dataset):
        labeleld_texts, labeled_labels = self.decoded_dataset(labeled_dataset)
        new_texts = []
        new_labels = []
        
        for idx in range(len(new_dataset)):
            decoded_text = new_dataset[idx][1]
            pred_label = new_dataset[idx][2]            
            
            new_texts.append(decoded_text)
            new_labels.append(pred_label)
        
        combined_texts = labeled_texts + new_texts
        combined_labels = labeled_labels + new_labels
        combined_dataset = self.encode_dataset(combined_texts, combined_labels)
        return combined_dataset, list(zip(new_texts, new_labels))
    
    
    def remove_dataset(self, unlabeled_dataset, new_dataset):
        unlabeled_texts = [data[0] for data in unlabeled_dataset]
        unlabeled_labels = [data[1] for data in unlabeled_dataset]
        
        new_texts = [data[0] for data in new_dataset]
        new_labels = [data[1] for data in new_dataset]
        
        # remove pseudo-labeled from unlabeled dataset
        for text in new_texts:
            idx = unlabeled_texts.index(text)
            unlabeled_texts.pop(idx)
            unlabeled_labels.pop(idx)
                    
        return list(zip(unlabeled_texts, unlabeled_labels))
    
        
    def encode_dataset(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        dataset = Dataset(encodings, labels)
        return dataset
    
    
    def decode_dataset(self, dataset):
        decoded_texts = []
        labels = []
        for idx in range(len(dataset)):
            text_id = dataset[idx]['input_ids']
            label = dataset[idx]['labels'].item()
            decoded_text = self.tokenizer.decode(text_id)
            decoded_text = decoded_text.replace("[CLS]", "").replace("[SEP]","").replace("[PAD]","").strip()
            decoded_texts.append(decoded_text)
            labels.append(label)
        return decoded_texts, labels
