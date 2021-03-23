import nltk
from nltk.corpus import opinion_lexicon
from nltk.stem import WordNetLemmatizer

def get_lexicon():
    opinion_pos = opinion_lexicon.positive()
    opinion_neg = opinion_lexicon.negative()
    lexicon = {0:opinion_neg, 1:opinion_pos}
    return lexicon

lexicon = get_lexicon()
lemmatizer = WordNetLemmatizer()

def guide_pseudo_labeling(pseudo_labeled, guide_type):
    '''
    @param dataset type: dict{pred_label}[list(tuple(text_id, decoded_text, pred_label, target, confidence))]
    '''     
    labels = pseudo_labeled.keys()
    new_dataset = {label: [] for label in labels}
    for label in labels:
        for data in pseudo_labeled[label]:
            text_id = data[0]
            decoded_text = data[1]
            model_pred = data[2]
            target = data[3]
            confidence = data[4]
            guide_pred = None
            
            if guide_type == 'predefined_lexicon':
                guide_pred = rule_base_with_lexicon(lexicon, decoded_text)
            elif guide_type == 'generated_lexicon':
                pass
            elif guide_type == 'naive_bayes':
                pass
            elif guide_type == 'advanced_nb':
                pass
            
            if model_pred == guide_pred:
                new_dataset[label].append((text_id, decoded_text, model_pred, target, confidence))    
    return new_dataset


def rule_base_with_lexicon(lexicon, text):
    ## To do: set threshold -> matching count
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
    
    if is_tie == True:
        predict_label = -1
    return predict_label
    