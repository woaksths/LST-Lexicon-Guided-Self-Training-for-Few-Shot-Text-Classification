from __future__ import print_function, division
import constant as config
import torch
import torchtext


class Evaluator(object):
    """ Class to evaluate models with given datasets.
    """
    def __init__(self, loss, batch_size=64):
        self.loss = loss
        self.batch_size = batch_size
        
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    def evaluate(self, model, data_loader, is_test=False):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model: model to evaluate
            data: dataset to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        loss = self.loss
        model.eval()
        device = config.device
        
        eval_loss = 0
        nb_eval_steps = 0
        nb_eval_examples = 0
        n_correct, n_total = 0, 0
        
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                ids = batch['input_ids'].to(device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                targets = batch['labels'].to(device, dtype=torch.long)
                
                outputs = model(ids, attention_mask, token_type_ids, labels=targets)
                loss, logits = outputs[0], outputs[1]
                
                eval_loss += loss.item()
                big_val, big_idx = torch.max(logits.data, dim=-1)
                n_correct += self.calculate_accu(big_idx, targets)
                
                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)
                
                if _ % 1000 == 0:
                    loss_step = eval_loss / nb_eval_steps
                    accu_step = (n_correct*100)/nb_eval_examples
                    if is_test == True:
                        print(f"Test Loss per 1000 steps: {loss_step}")
                        print(f"Test Accuracy per 1000 steps: {accu_step}")
                    else:
                        print(f"Validation Loss per 1000 steps: {loss_step}")
                        print(f"Validation Accuracy per 1000 steps: {accu_step}")
            
        epoch_loss = eval_loss / nb_eval_steps
        epoch_accu = (n_correct*100) / nb_eval_examples
        
        if is_test == True:
            print(f"Test Loss Epoch: {epoch_loss}")
            print(f"Test Accuracy Epoch: {epoch_accu}")
        else:
            print(f"Validation Loss Epoch: {epoch_loss}")
            print(f"Validation Accuracy Epoch: {epoch_accu}")
        return epoch_loss, epoch_accu
