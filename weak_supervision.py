import nltk
from nltk.corpus import opinion_lexicon
from nltk.stem import WordNetLemmatizer

def get_lexicon():
    opinion_pos = opinion_lexicon.positive()
    opinion_neg = opinion_lexicon.negative()
    lexicon = {0:opinion_neg, 1:opinion_pos}
    return lexicon

lemmatizer = WordNetLemmatizer()

def guide_pseudo_labeling(pseudo_labeled, guide_type, lexicon=None):
    '''
    @param dataset type: dict{pred_label}[list(tuple(text_id, decoded_text, pred_label, target, confidence))]
    '''
    labels = pseudo_labeled.keys()
    new_dataset = {label: [] for label in labels}
    print('#'*100)
#     print(guide_type)
#     print(lexicon)
    
    for label in labels:
        for data in pseudo_labeled[label]:
            text_id = data[0]
            decoded_text = data[1]
            model_pred = data[2]
            target = data[3]
            confidence = data[4]
            guide_pred = None
            
            if guide_type == 'predefined_lexicon':
                lexicon = get_lexicon()
                guide_pred = rule_base_with_lexicon(lexicon, decoded_text)
            elif guide_type == 'generated_lexicon':
                guide_pred = rule_base_with_lexicon(lexicon, decoded_text)
            elif guide_type == 'naive_bayes':
                pass
            elif guide_type == 'advanced_nb':
                pass
            
            if model_pred == guide_pred:
                new_dataset[label].append((text_id, decoded_text, model_pred, target, confidence))    
    return new_dataset

'''
1. 렉시콘 매칭 시, 같은 단어가 서로 다른 클래스에 등장한다면 빈도수로 가장 많이 받은 빈도수의 매칭을 택하여 해당 레이블의 카운트를 늘려줌
2. 렉시콘 stop_words 제거 
'''

def rule_base_with_lexicon(lexicon, text):
    
    words = text.split(' ')
    words = [lemmatizer.lemmatize(word) for word in words]
    labels = lexicon.keys()
    num_of_matching = {label:0 for label in labels}
    
    for word in words:
        for label in labels:
            if word in lexicon[label]:
                num_of_matching[label] +=1
    
    predict_label = -1
    max_count = -1
    is_tie = False
    for label, count in num_of_matching.items():
        if count > max_count:
            max_count = count
            predict_label = label
        elif count == max_count:
            is_tie = True
    
    ## To do: set threshold -> matching count
    if is_tie == True and max_count <= 2:
        predict_label = -1
    return predict_label
    