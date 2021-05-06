Lexicon Construction
-----------------------------------------
<p align="center">
<img src="/asset/lexicon_construction.png" width="60%" height="40%"></img>
</p>

We adopt the pretrained language model, BERT, as a base model to classic self-training since contextual word embeddings of pre-trained language model has tremendously reduced the need for manual data annotation. We also extend the traditional self-training to use a lexicon, which is a set of representative words for each class in a specific domain.
The lexicon, crafted by frequent occurrence of semantically important words, enhances the reliability of the pseudo-labeling mechanism as role of weak supervision. Our lexicon construction is as follows. 

1. <b>Construction</b>

- The [CLS] token is a special token that appears before the input sequence in the BERT. The [CLS] token is learned as a hidden vector that reflects the entire context of the input sequence. Since the hidden vector corresponding to the [CLS] contains the context of the entire sequence, the vector is used as an input to the classifier.


2. <b>Refinement</b>
