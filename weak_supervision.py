import nltk
from nltk.corpus import opinion_lexicon
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def get_lexicon():
    # This lexicon must be used for sentiment classification (opinion).
    # Opinion Lexicon (or Sentiment Lexicon) (Hu and Liu, KDD-2004).
    opinion_pos = opinion_lexicon.positive()
    opinion_neg = opinion_lexicon.negative()
    lexicon = {0:opinion_neg, 1:opinion_pos} 
    return lexicon



def guide_pseudo_labeling(pseudo_labeled, guide_type, lexicon=None):
    '''
    @param dataset type: dict(label:[tuple(text_id, decoded_text, pred_label, target, confidence)])
    '''
    new_dataset = None
    if guide_type == 'predefined_lexicon_pl':
        lexicon = get_lexicon()
        new_dataset = bow_PL(pseudo_labeled, lexicon)
    elif guide_type == 'lexicon_pl':
        new_dataset = lexicon_PL(pseudo_labeled, lexicon)
    elif guide_type == 'weighted_lexicon_pl':
        new_dataset = weighted_lexicon_PL(pseudo_labeled, lexicon)
    return new_dataset



def bow_PL(pseudo_labeled, lexicon):
    labels = pseduo_labeled.keys()
    new_dataset = {label:[] for label in labels}
    
    for label in labels:
        for data in pseudo_labeled[label]:
            text_id = data[0]
            decoded_text = data[1]
            model_pred = data[2]
            target = data[3]
            confidence = data[4]
            
            words = decoded_text.split(' ')
            words = [lemmatizer.lemmatize(word) for word in words]
            num_of_matching = {label:0 for label in labels}
            
            for word in words:
                for index in labels:
                    if word in lexicon[index]:
                        num_of_matching[index] += 1
            
            lexicon_pred = -1
            max_count = -1
            is_tie = False
            for index, count in num_of_matching.items():
                if count > max_count:
                    max_count = count
                    lexicon_pred = index
                    is_tie = False
                elif count == max_count:
                    is_tie = True
            
            if is_tie == True or max_count <2:
                lexicon_pred = -1
            if model_pred == lexicon_pred:
                new_dataset[label].append((text_id, decoded_text, model_pred, target, confidence))
    return new_dataset



def lexicon_PL(pseudo_labeled, lexicon):
    labels = pseudo_labeled.keys()
    new_dataset = {label: [] for label in labels}
    
    for label in labels:
        for data in pseudo_labeled[label]:
            text_id = data[0]
            decoded_text = data[1]
            model_pred = data[2]
            target = data[3]
            confidence = data[4]
            words = decoded_text.split(' ')
            words = [lemmatizer.lemmatize(word) for word in words]
            num_of_matching = {label:0 for label in labels}
            
            for word in words:
                word_label = None
                max_count = 0
                is_tie = False
                for index in list(lexicon.keys()):
                    if word in lexicon[index]:
                        count = lexicon[index][word]
                        if max_count < count:
                            max_count = count
                            is_tie = False
                            word_label = index
                        elif max_count == count:
                            is_tie = True
                if is_tie is False and word_label is not None:
                    num_of_matching[word_label] += 1 

            lexicon_pred = -1 
            max_count = 0
            is_tie = False
            for index, count in num_of_matching.items():
                if count > max_count:
                    max_count = count
                    lexicon_pred = index
                    is_tie = False
                elif count == max_count:
                    is_tie = True
            if is_tie == True or max_count < 2:
                lexicon_pred = -1
                
            if model_pred == lexicon_pred:
                new_dataset[label].append((text_id, decoded_text, model_pred, target, confidence))
    return new_dataset
