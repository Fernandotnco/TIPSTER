import torch
import torch.nn as nn

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class ExplainedVariance:
    """
    Computes:
        EV_sem  = 1 - MSE(Z, S)      / Var(Z)
        EV_full = 1 - MSE(Z, S+R)    / Var(Z)

    Use:
        ev = ExplainedVariance()
        ev_sem, ev_full = ev(Z, S, R)
    """
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, Z, S, R):
        # Z: [B, D]
        # S: [B, D]
        # R: [B, D]

        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        varZ = (Z_centered ** 2).mean() + self.eps

        mse_sem  = ((Z - S)**2).mean()
        mse_full = ((Z - (S + R))**2).mean()

        EV_sem  = 1 - mse_sem  / varZ
        EV_full = 1 - mse_full / varZ

        return EV_sem.item(), EV_full.item()

class MRRMetric:
    """
    Computes MRR for (anchor, positive) retrieval.
    Use:
        mrr = MRRMetric()
        score = mrr(anchors, positives)
    """
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, A, P):
        # A, P: [B, D]
        A_norm = A / (A.norm(dim=1, keepdim=True) + self.eps)
        P_norm = P / (P.norm(dim=1, keepdim=True) + self.eps)

        # Similarity matrix: sim[i, j] = cos(A_i, P_j)
        sim = A_norm @ P_norm.T  # [B, B]

        # For each anchor i, the positive is sim[i, i]
        pos_scores = sim.diag().unsqueeze(1)    # [B, 1]

        # Rank positives by comparing with all scores for same anchor
        ranks = (sim >= pos_scores).sum(dim=1)

        # Reciprocal rank
        rr = 1.0 / ranks.float()

        # Mean Reciprocal Rank
        return rr.mean().item()

class DecoderGap:
    """
    Measures similarity between:
        Dec(Z)            == AE reconstruction
        Dec(S_pos + R)    == decomposition reconstruction

    By default uses MSE. If "similarity=True", returns (1 - MSE).
    """
    def __init__(self, similarity=False, eps=1e-8):
        self.similarity = similarity
        self.eps = eps

    def __call__(self, X_ae, X_sr):
        # both tensors: [B, C, H, W] or [B, T] for audio
        mse = ((X_ae - X_sr)**2).mean()

        if self.similarity:
            # return similarity score in [0, 1]
            return max(0.0, 1.0 - mse.item())
        else:
            # return raw MSE gap
            return mse.item()




# ================================================================
#   1. SemanticProbeMetric
#      - trains logistic/ridge probes for head / complement / full Z
#      - computes normalized P_head, P_comp, D_k, E_k
# ================================================================
class SemanticProbeMetric:
    def __init__(self, semantic_info):
        """
        semantic_info: dict of:
           {
               "semantic_name": {
                   "head_index": int,
                   "type": "classification" | "regression" | "none"
               }
           }
        """
        self.semantic_info = semantic_info

        # Storage for training embeddings
        # train_storage[sem]["head"|"comp"|"full"] = [numpy arrays]
        self.train_storage = {
            sem: {"head": [], "comp": [], "full": [], "y": []}
            for sem, info in semantic_info.items()
            if info.get("type", "classification") != "none"
        }

        # Probes after fitting
        self.probes = {}  # sem_name -> dict with {"head": model, "comp": model, "full": model}

    # ------------------------------------------------------------
    # For training we accumulate features; fitting happens later.
    # ------------------------------------------------------------
    def update_train(self, sem_name, head_embeddings, complement_embeddings, full_embeddings, labels, sem_type):
        if sem_type == "none":
            return

        deh = head_embeddings.detach().cpu().numpy()
        dec = complement_embeddings.detach().cpu().numpy()
        dez = full_embeddings.detach().cpu().numpy()

        y = labels.detach().cpu().numpy()

        store = self.train_storage[sem_name]
        store["head"].append(deh)
        store["comp"].append(dec)
        store["full"].append(dez)
        store["y"].append(y)

    # ------------------------------------------------------------
    # Fit the probes for all semantic factors
    # ------------------------------------------------------------
    def fit(self):
        for sem, info in self.semantic_info.items():
            sem_type = info.get("type", "classification")
            if sem_type == "none":
                continue

            store = self.train_storage[sem]
            Xh = np.concatenate(store["head"], axis=0)
            Xc = np.concatenate(store["comp"], axis=0)
            Xz = np.concatenate(store["full"], axis=0)
            y  = np.concatenate(store["y"],   axis=0)

            if sem_type == "classification":
                model_head = LogisticRegression(max_iter=300)
                model_comp = LogisticRegression(max_iter=300)
                model_full = LogisticRegression(max_iter=300)

            else:  # regression
                model_head = Ridge(alpha=1.0)
                model_comp = Ridge(alpha=1.0)
                model_full = Ridge(alpha=1.0)

            model_head.fit(Xh, y)
            model_comp.fit(Xc, y)
            model_full.fit(Xz, y)


            self.probes[sem] = {
                "head": model_head,
                "comp": model_comp,
                "full": model_full,
                "type": sem_type
            }

    # ------------------------------------------------------------
    # Evaluate one batch of embeddings
    # ------------------------------------------------------------
    def evaluate_batch(self, sem_name, head_embeddings, complement_embeddings, full_embeddings, labels, sem_type):
        """
        Returns:
            dict of {f"{sem_name}_Phead": ..., f"{sem_name}_Pcomp": ..., f"{sem_name}_D": ..., f"{sem_name}_E": ...}
        """
        if sem_type == "none":
            return {}

        model_head = self.probes[sem_name]["head"]
        model_comp = self.probes[sem_name]["comp"]
        model_full = self.probes[sem_name]["full"]

        Xh = head_embeddings.detach().cpu().numpy()
        Xc = complement_embeddings.detach().cpu().numpy()
        Xz = full_embeddings.detach().cpu().numpy()
        y  = labels.detach().cpu().numpy()

        # Predict
        if sem_type == "classification":
            pred_h = model_head.predict(Xh)
            pred_c = model_comp.predict(Xc)

            P_head = accuracy_score(y, pred_h)
            P_comp = accuracy_score(y, pred_c)

            # Baseline chance = 1 / num_classes (assumed from training)
            # If unknown: fallback to uniform chance
            classes = np.unique(y)
            chance = 1.0 / len(classes)

            # Avoid division by zero
            norm_P_head = (P_head - chance) / (1 - chance + 1e-8)
            norm_P_comp = (P_comp - chance) / (1 - chance + 1e-8)

        else:  # regression
            pred_h = model_head.predict(Xh)
            pred_c = model_comp.predict(Xc)

            R2_h = max(0.0, r2_score(y, pred_h))  # clip negative
            R2_c = max(0.0, r2_score(y, pred_c))

            norm_P_head = R2_h
            norm_P_comp = R2_c

        # Combined metrics
        D_k = norm_P_head - norm_P_comp
        E_k = norm_P_head * (1 - norm_P_comp)

        return {
            f"{sem_name}_P_head": float(norm_P_head),
            f"{sem_name}_P_comp": float(norm_P_comp),
            f"{sem_name}_D": float(D_k),
            f"{sem_name}_E": float(E_k),
        }


# ================================================================
# 2. CrossHeadPredictabilityMetric
#    - collects all s_k per batch
#    - at final: fits ridge on s_i → s_j
#    - produces R² matrix
# ================================================================
class CrossHeadPredictabilityMetric:
    def __init__(self, semantic_info):
        self.semantic_info = semantic_info
        self.storage = []  # list of lists: [S_A], each S_A is list of heads: [B,D]

    def update(self, S_A):
        """
        S_A: list of semantic embeddings [ [B,D], [B,D], ... ]
        """
        # convert to numpy
        H = []
        for s in S_A:
            H.append(s.detach().cpu().numpy())
        self.storage.append(H)

    def compute(self):
        if not self.storage:
            return {}

        # stack batches
        H_all = list(zip(*self.storage))  # group by head index
        H_all = [np.concatenate(hlist, axis=0) for hlist in H_all]  # each [N,D]

        num_heads = len(H_all)
        if num_heads == 1:
            return {
            "cross_R2_matrix": 1,
            "cross_redundancy": 0,
        }
        R2_matrix = np.zeros((num_heads, num_heads), dtype=np.float32)

        for i in range(num_heads):
            Xi = H_all[i]
            for j in range(num_heads):
                if i == j:
                    R2_matrix[i, j] = 1.0
                    continue
                Yj = H_all[j]

                reg = Ridge(alpha=1.0)
                reg.fit(Xi, Yj)
                pred = reg.predict(Xi)

                # mean R2 over all dims
                r2 = 0
                for d in range(Yj.shape[1]):
                    r2 += r2_score(Yj[:, d], pred[:, d])
                r2 /= Yj.shape[1]
                R2_matrix[i, j] = max(0.0, r2)

        # off-diagonal mean
        offdiag = []
        for i in range(num_heads):
            for j in range(num_heads):
                if i != j:
                    offdiag.append(R2_matrix[i, j])
        cross_redundancy = float(np.mean(offdiag))

        return {
            "cross_R2_matrix": R2_matrix,
            "cross_redundancy": cross_redundancy,
        }


# ================================================================
# 3. DCIHeadImportanceMetric
#    (similar to DCI score: uses feature importances from RF)
# ================================================================
class DCIHeadImportanceMetric:
    def __init__(self, semantic_info):
        self.semantic_info = semantic_info

        # For each semantic factor, store X = concatenated heads, Y label
        self.storage = {
            sem: {"X": [], "y": []}
            for sem, info in semantic_info.items()
            if info.get("type", "classification") != "none"
        }

    def update(self, S_A, labels):
        """
        S_A: list of head embeddings [B,D] for anchor
        labels: dict {sem_name: tensor[B]}
        """
        if not self.storage:
            return

        # concatenate heads horizontally → [B, D_total]
        H_np = [s.detach().cpu().numpy() for s in S_A]
        X = np.concatenate(H_np, axis=1)

        for sem, data in self.storage.items():
            y = labels[sem].detach().cpu().numpy()
            data["X"].append(X)
            data["y"].append(y)

    def compute(self):
        if not self.storage:
            return {}

        importance_matrix = {}
        head_alignment = {}
        head_completeness = {}

        # Number of heads
        head_count = None

        for sem, d in self.storage.items():
            X = np.concatenate(d["X"], axis=0)
            y = np.concatenate(d["y"], axis=0)

            # figure out number of heads
            if head_count is None:
                # assume fixed dimension: D_total / each head dimension
                # too hard to infer reliably without info – we assume
                # each head embedding has same dimension
                pass

            info = self.semantic_info[sem]
            sem_type = info.get("type", "classification")
            if sem_type == "classification":
                rf = RandomForestClassifier(n_estimators=50)
            else:
                rf = RandomForestRegressor(n_estimators=50)

            rf.fit(X, y)
            importances = rf.feature_importances_  # [D_total]

            # Now slice importances by heads
            # Only safe if all heads have same dimension
            # For now we assume that’s the case (your design).
            D_total = importances.shape[0]
            num_heads = len(self.semantic_info)
            head_dim = D_total // num_heads

            imp_per_head = []
            for i in range(num_heads):
                start = i * head_dim
                end = (i + 1) * head_dim
                imp_per_head.append(np.sum(importances[start:end]))

            imp_per_head = np.array(imp_per_head)
            imp_per_head /= (imp_per_head.sum() + 1e-8)  # normalize

            importance_matrix[sem] = imp_per_head

            # alignment = max head gets most importance
            head_alignment[sem] = float(np.max(imp_per_head))

            # completeness = how uniformly importance is distributed
            head_completeness[sem] = float(1.0 - np.std(imp_per_head))

        return {
            "importance_matrix": importance_matrix,
            "head_alignment": head_alignment,
            "head_completeness": head_completeness,
        }


# ================================================================
# 4. ReconstructionDiffMetric
#    - accumulate difference stats for partial reconstructions
# ================================================================
class ReconstructionDiffMetric:
    def __init__(self):
        # tag -> list of difference magnitudes
        self.storage = {}

    def update(self, full_recon, partial_recon, tag):
        """
        full_recon, partial_recon: [B,C,H,W]
        Compute difference e.g. L1 or L2 – but exact metric is abstract here.
        We only store per-pixel absolute diff mean as reference.
        You can replace this with your own fidelity metric later.
        """
        diff = (full_recon - partial_recon).abs().mean(dim=(1, 2, 3))  # [B]
        diff = diff.detach().cpu().numpy()

        if tag not in self.storage:
            self.storage[tag] = []

        self.storage[tag].append(diff)

    def compute(self):
        out = {}
        for tag, vals in self.storage.items():
            arr = np.concatenate(vals, axis=0)
            out[f"recon_diff_{tag}"] = float(arr.mean())
        return out
