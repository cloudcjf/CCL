import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchpack.utils.config import configs 
from tqdm import tqdm 
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity


class ContrastiveLoss(torch.nn.Module):
    # from paper the optimal temperature is 0.5
    def __init__(self, temperature=0.5):
        torch.nn.Module.__init__(self)
        self.temperature = temperature
        # self.distance = LpDistance(normalize_embeddings=True, collect_stats=True)
        self.distance = CosineSimilarity(collect_stats=True)

    """
        anchor size: [1, output_dim]
        positive size: [1, output_dim]
        negatives size: [batch-2, output_dim]
    """
    def forward(self, embeddings, positives_mask, negatives_mask):
        similarity_mat = self.distance(embeddings)  # [batch_size, batch_size]
        batch_loss = []
        for i in range(similarity_mat.shape[0]):
            row_similarity = similarity_mat[i]
            row_positive_mask = positives_mask[i]
            row_negative_mask = negatives_mask[i]
            # skip the row that has no positive or negative pairs
            if torch.sum(row_positive_mask) == 0 or torch.sum(row_negative_mask) == 0:
                continue
            positive = row_similarity[row_positive_mask]
            negative = row_similarity[row_negative_mask]
            positive /= self.temperature
            positive = torch.exp(positive)
            negative /= self.temperature
            negative = torch.exp(negative)
            each = torch.mean(positive)
            batch_loss.append(-torch.log(each / (torch.sum(negative) + each)))
        return sum(batch_loss) / similarity_mat.shape[0]