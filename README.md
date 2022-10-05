# Unsupervised-continual-place-recognition
## Problem definition
1. Joint training is expensive and sometimes impossible.
2. Massive data comes in sequence in the real world, we are unable to access previous data or all sequences at the same time.
3. When retraining a model on a different domain sequentially, catastrophic forgetting will be caused.
4. Labeling all the data is expensive and sometimes impossible.

## Baseline
**InCloud** has proposed a continual learning strategy for the sequential place recognition task. This strategy efficiently eliminates catastrophic forgetting. 

We plan to try unsupervised learning on unlabeled data in this sequential training process to avoid the expensive labeling process.

## Datasets
- Oxford
- Inhouse
- Mulran
- KITTI

## Place recognition methods
- MinkLoc3D
- Logg3d
- PointNetVLAD

## Experiments
### Step 1
Train the model on the Oxford dataset.

Use the pre-trained model to infer global descriptors on the Inhouse dataset.

Generate pseudo labels (query, positive, negative) according to the distances between descriptors.
Train on the new dataset with pseudo-labeled tuples.

### Results
**no_retrain**: Supervised training on Oxford dataset, evaluation on Inhouse dataset directly.

**retrain_wo_cl**: Supervised training on Oxford dataset, supervised retrain on Inhouse dataset, no continual learning strategy is implemented.

**retrain_wi_cl**: Supervised training on Oxford dataset, supervised retrain on Inhouse dataset, continual learning strategy is implemented.

**joint_train**: Supervised training on Oxford dataset and Inhouse dataset jointly.

**unsuper_retrain_wo_cl**: our method, Supervised training on Oxford dataset, unsupervised retrain on Inhouse dataset, no continual learning strategy is implemented.

**unsuper_retrain_wi_cl**: our method, Supervised training on Oxford dataset, unsupervised retrain on Inhouse dataset, continual learning strategy is implemented.

| Recall@1 | Oxford | Business | Resident | University |
|----|---|---|---|---|
| no_retrain | 93.8 | 82.7 | 81.1 | 96.0 |
| retrain_wo_cl | 70.3 | 93.3 | 96.4 | 96.3 |
| retrain_wi_cl | 90.7 | 93.3 | 95.8 | 96.1 |
| joint_train | 94.8 | 94.0 | 96.7 | 97.2 |
| unsuper_retrain_wo_cl | 78.8 | 85.3 | 84.8 | 86.4 |
| unsuper_retrain_wi_cl | 90.0 | 86.1 | 88.2 | 90.8 |