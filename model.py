## 4. 렉시콘은 best dev_acc일 때 생성 (cls, pad, sep, stop_words 제외 단어집 생성 count 고려해서 생성) 문장 내 중복 단어 binary할지


import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from transformers import BertTokenizer

class BERT_ATTN(torch.nn.Module):
    
    def __init__(self, num_labels):
        super(BERT_ATTN, self).__init__()

        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.num_labels)
        # self.init_weights()
        self.attention_mask = None
    
    def attention(self, pooler_output, last_hidden_state):
        pooler_output = pooler_output.unsqueeze(1)
        last_hidden_state = last_hidden_state.transpose(1,2)

        attn = torch.bmm(pooler_output, last_hidden_state)
        attn = attn.squeeze(1)
        
        if self.attention_mask is not None:
            mask = (1-self.attention_mask).type(torch.BoolTensor).to('cuda')
            attn.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn, dim=-1)
        
        return attn
    
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        last_hidden_state, pooler_output = self.bert(input_ids=input_ids, 
                                                     attention_mask=attention_mask,token_type_ids=token_type_ids)
        
        # set attention mask
        self.attention_mask = attention_mask
        self.attention_mask[:, 0] = 0 # mask cls token 
        sep_mask = (input_ids == self.tokenizer.sep_token_id) # mask sep token
        self.attention_mask.masked_fill_(sep_mask, 0)
        
        attn = self.attention(pooler_output, last_hidden_state)
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits, attn)
        
