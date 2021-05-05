# Weak-Supervision-Based-Self-Training

**LST** or **L**exicon-guided **S**elf-**T**raining is a method of task-specific training of pre-trained language model with only a few labeled examples for the target classification task. **LST** is based on the self-training framework and trained with 4 simple steps:

1. Train a classifier on labeled data (teacher) and [build the lexicon](/asset/lexicon_construction.md) with attention mechanism.
2. Infer labels on a much larger unlabeled dataset with [lexicon-guided pseudo-labeling](/asset/lexicon_guided_PL.md). 
3. Train a larger classifier (student) on combined set with noise and [build the lexicon](/asset/lexicon_construction.md) from student. 
4. Go to step 2, with studnet as teacher. And update the lexicon with the ones built at student steps.

![](/asset/LST_overview.png)
