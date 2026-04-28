import torch
import torch.nn as nn
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.5):
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()

def ev_loss(s, z, eps=1e-8):
    s_hat = F.normalize(s, p=2, dim=-1, eps=eps)
    proj = (z * s_hat).sum(-1, keepdim=True) * s_hat
    resid = z - proj
    return (resid.pow(2).sum(-1) / (z.pow(2).sum(-1) + eps)).mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor   = F.normalize(anchor,   p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)

        return F.relu(d_pos - d_neg + self.margin).mean()
    

class SmoothTripletLoss(nn.Module):
    def __init__(self, margin=0.5, tau = 0.05):
        super().__init__()
        self.margin = margin
        self.tau=tau
        self.softplus = nn.Softplus()

    def forward(self, anchor, positive, negative):
        anchor   = F.normalize(anchor,   p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)

        return self.softplus((d_pos - d_neg + self.margin) / self.tau).mean()



class EVLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, s, z):
        eps = self.eps

        s_hat = F.normalize(s, p=2, dim=-1, eps=eps)

        proj = (z * s_hat).sum(dim=-1, keepdim=True) * s_hat
        resid = z - proj

        num = resid.pow(2).sum(dim=-1)
        den = z.pow(2).sum(dim=-1) + eps

        return (num / den).mean()



class SoftNearestNeighborLoss(nn.Module):
    """
    Standard Soft Nearest Neighbor Loss (Frosst et al. 2019).
    Assumes a single categorical label per sample.
    """
    def __init__(self, temperature: float = 0.5, similarity: str = "cosine"):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [N, D]
        labels: [N] (int tensor)
        """
        device = embeddings.device
        N = embeddings.size(0)

        # Compute pairwise similarity matrix
        if self.similarity == "cosine":
            emb = F.normalize(embeddings, dim=1)
            sim = emb @ emb.T  # [N, N]
        elif self.similarity == "dot":
            sim = embeddings @ embeddings.T
        else:
            raise ValueError(f"Unsupported similarity: {self.similarity}")

        # Remove self-similarity
        mask = torch.eye(N, device=device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Compute label equality mask
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
        label_eq &= ~mask

        # Numerator: same-class similarities
        exp_sim = torch.exp(sim / self.temperature)
        num = (exp_sim * label_eq.float()).sum(dim=1)

        # Denominator: all similarities
        denom = exp_sim.sum(dim=1) + 1e-8

        loss = -torch.log(num / denom + 1e-8)
        return loss.mean()


class MultiLabelSoftNearestNeighborLoss(nn.Module):
    """
    Soft Nearest Neighbor Loss for multi-hot label vectors.
    Encourages embeddings to reflect partial label overlaps.
    """
    def __init__(self, temperature: float = 0.5, similarity: str = "cosine", label_similarity: str = "cosine"):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
        self.label_similarity = label_similarity

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [N, D]
        labels: [N, A] (float multi-hot vectors, values ∈ {0,1})
        """
        device = embeddings.device
        N = embeddings.size(0)

        # --- Embedding similarity ---
        if self.similarity == "cosine":
            emb = F.normalize(embeddings, dim=1)
            sim_emb = emb @ emb.T
        elif self.similarity == "dot":
            sim_emb = embeddings @ embeddings.T
        else:
            raise ValueError(f"Unsupported similarity: {self.similarity}")

        # --- Label similarity ---
        if self.label_similarity == "cosine":
            lab = F.normalize(labels.float(), dim=1)
            sim_lab = lab @ lab.T
        elif self.label_similarity == "overlap":
            # Jaccard-style overlap: |A∩B| / |A∪B|
            inter = (labels.unsqueeze(1) * labels.unsqueeze(0)).sum(-1)
            union = ((labels.unsqueeze(1) + labels.unsqueeze(0)) > 0).float().sum(-1) + 1e-8
            sim_lab = inter / union
        else:
            raise ValueError(f"Unsupported label_similarity: {self.label_similarity}")

        # --- Exclude self-comparisons ---
        mask = torch.eye(N, device=device, dtype=torch.bool)
        sim_emb.masked_fill_(mask, -1e9)
        sim_lab.masked_fill_(mask, 0.0)

        # --- Weighted SNN numerator/denominator ---
        exp_sim = torch.exp(sim_emb / self.temperature)
        num = (exp_sim * sim_lab).sum(dim=1)
        denom = exp_sim.sum(dim=1) + 1e-8

        loss = -torch.log(num / denom + 1e-8)
        return loss.mean()

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)
    Expects features of shape [batch_size, n_views, dim]
    and labels of shape [batch_size].
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, V, D]
        device = features.device
        features = features.unsqueeze(1)
        bsz, n_views = features.shape[:2]

        # L2 normalize
        features = F.normalize(features, dim=-1)

        # Flatten views: [B * V, D]
        feats = features.reshape(bsz * n_views, -1)

        # Similarity matrix: [BV, BV]
        logits = torch.div(
            feats @ feats.t(),
            self.temperature
        )

        # Mask out self-comparison
        logits_mask = torch.ones_like(logits, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        # Build positive mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).to(device)   # [B, B]

        # Expand mask for views: [BV, BV]
        mask = mask.repeat_interleave(n_views, dim=0).repeat_interleave(n_views, dim=1)
        mask = mask & logits_mask  # remove anchor itself

        # For each anchor, denominator excludes itself
        exp_logits = torch.exp(logits) * logits_mask

        # For each anchor:
        #   sum over its positive views (numerator)
        #   sum over all others (denominator)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Only keep positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Final loss
        loss = -mean_log_prob_pos.view(bsz, n_views).mean()
        return loss

class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, lam=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

    def forward(self, embeddings, labels):
        # embeddings: [B, D], labels: [B]
        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = embeddings @ embeddings.t()
        
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.t()).float()
        mask_neg = 1 - mask_pos
        
        # exclude self-pairs
        mask_pos -= torch.eye(mask_pos.size(0), device=mask_pos.device)

        s = sim_matrix
        
        # Positive part
        pos_term = torch.log1p(torch.sum(
            torch.exp(-self.alpha * (s - self.lam)) * mask_pos, dim=1
        )) / self.alpha
        
        # Negative part
        neg_term = torch.log1p(torch.sum(
            torch.exp(self.beta * (s - self.lam)) * mask_neg, dim=1
        )) / self.beta
        
        loss = pos_term + neg_term
        return loss.mean()

class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, emb_dim, alpha=32, margin=0.1):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        
        self.proxies = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        proxies = F.normalize(self.proxies, dim=1)

        sim = embeddings @ proxies.t()  # [B, C]

        device = embeddings.device
        labels_onehot = F.one_hot(labels, num_classes=proxies.size(0)).float().to(device)

        pos_mask = labels_onehot.bool()
        neg_mask = ~pos_mask

        pos_exp = torch.exp(-self.alpha * (sim - self.margin)) * pos_mask
        neg_exp = torch.exp(self.alpha * (sim - self.margin)) * neg_mask

        # aggregate over classes that appear
        pos_loss = torch.log1p(pos_exp.sum(dim=0))
        neg_loss = torch.log1p(neg_exp.sum(dim=0))

        num_pos_classes = pos_mask.sum(dim=0).clamp(min=1)
        
        loss = (pos_loss / num_pos_classes + neg_loss).mean()
        return loss
    
class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=5e-3):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a, z_p, z_n=None):
        
        batch_size = z_a.size(0)
        feature_dim = z_a.size(1)


        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)
        z_p_norm = (z_p - z_p.mean(0)) / z_p.std(0)


        c = torch.mm(z_a_norm.T, z_p_norm) / batch_size


        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        

        off_diag = c.flatten()[:-1].view(feature_dim - 1, feature_dim + 1)[:, 1:].flatten()
        off_diag = off_diag.pow_(2).sum()

        loss = on_diag + self.lambda_param * off_diag
        return loss
