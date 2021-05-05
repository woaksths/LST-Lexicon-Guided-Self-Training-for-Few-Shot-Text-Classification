# Weak-Supervision-Based-Self-Training

**LST** or **L**exicon-guided **S**elf-**T**raining is a method of task-specific training of pre-trained language model with only a few labeled examples for the target classification task. **LST** is based on the self-training framework and trained with 5 simple steps:

1. Train a classifier on labeled data (teacher) and build the lexicon with attention mechanism.
2. Infer labels on a much larger unlabeled dataset with lexicon-guided pseudo-labeling. 
3. Train a larger classifier (student) on combined set, adding noise and builde the lexicon from student. 
4. Go to step 2, with studnet as teacher and update the lexicon with the ones built at student steps.
