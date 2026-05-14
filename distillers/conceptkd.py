import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ._base import BaseDistiller
from .registry import register_distiller


def normalize_mean_std(x, eps=1e-6):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + eps)


def l2_dist(x, p):
    return torch.cdist(x, p, p=2).pow(2) / x.shape[-1]


def sparsify_mapping(A, topk=0, eps=1e-12):
    if topk is None or topk <= 0 or topk >= A.shape[-1]:
        return A

    vals, idx = torch.topk(A, k=topk, dim=-1)
    A_sparse = torch.zeros_like(A)
    A_sparse.scatter_(dim=-1, index=idx, src=vals)
    A_sparse = A_sparse / (A_sparse.sum(dim=-1, keepdim=True) + eps)
    return A_sparse

def kd_loss(logits_student, logits_teacher, temperature=1.0):
    log_p_s = F.log_softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)

@torch.no_grad()
def distributed_sinkhorn_probs(P, nmb_iters=3, sharpen=2.0, world_size=1, eps=1e-12):
    """
    P: [1, N, K], already-softmaxed concept probabilities.
    Returns balanced assignments with same shape.
    """
    dtype = P.dtype
    P = P.float()

    if sharpen != 1.0:
        P = P.pow(sharpen)

    Q = P.permute(0, 2, 1).contiguous()  # [1, K, N]

    sum_Q = Q.sum(dim=(1, 2), keepdim=True)
    if world_size > 1:
        dist.all_reduce(sum_Q)
    Q = Q / (sum_Q + eps)

    B = Q.shape[2] * world_size
    K = Q.shape[1]

    for _ in range(nmb_iters):
        row_sum = Q.sum(dim=2, keepdim=True)
        if world_size > 1:
            dist.all_reduce(row_sum)
        Q = Q / (row_sum + eps)
        Q = Q / K

        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
        Q = Q / B

    Q = Q * B
    return Q.permute(0, 2, 1).contiguous().to(dtype)  # [1, N, K]


class Prototypes(nn.Module):
    def __init__(self, num_prototypes, dim):
        super().__init__()
        self.protos = nn.Parameter(torch.randn(1, num_prototypes, dim) * 0.02)


@register_distiller
class ConceptKD(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super().__init__(student, teacher, criterion, args)

        assert len(args.concept_stages) == len(args.concept_mapping_stages), (
            "concept_stages and concept_mapping_stages must have the same length"
        )
        assert len(args.concept_stages) == len(args.concept_mapping_temps), (
            "concept_stages and concept_mapping_temps must have the same length"
        )

        self.stages = args.concept_stages
        self.stage_keys = [str(s) for s in self.stages]

        self.projector_s = nn.ModuleDict()
        self.projector_t = nn.ModuleDict()
        self.prototypes = nn.ModuleDict()

        payload = torch.load(args.concept_mapping_path, map_location="cpu")

        for stage, map_stage, map_temp, stage_key in zip(
            args.concept_stages,
            args.concept_mapping_stages,
            args.concept_mapping_temps,
            self.stage_keys,
        ):
            _, shape_s = student.stage_info(stage)
            _, shape_t = teacher.stage_info(stage)

            dim_s = shape_s[-1] if len(shape_s) == 2 else shape_s[0]
            dim_t = shape_t[-1] if len(shape_t) == 2 else shape_t[0]

            self.projector_s[stage_key] = nn.Linear(dim_s, args.concept_dim)
            self.projector_t[stage_key] = nn.Linear(dim_t, args.concept_dim)
            self.prototypes[stage_key] = Prototypes(args.concept_num_prototypes, args.concept_dim)

            mapping_logits = payload["stages"][map_stage]["mapping_logits"].float()
            mapping_probs = torch.softmax(mapping_logits / map_temp, dim=-1)
            mapping_probs = sparsify_mapping(mapping_probs, topk=args.concept_mapping_topk)

            self.register_buffer(f"mapping_probs_{stage_key}", mapping_probs)  # [196, T]

    def _local_tokens(self, feat):
        if feat.dim() == 4:
            return feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        if feat.dim() == 3:
            return feat[:, 1:, :]  # [B, 196, C]
        raise RuntimeError(f"Unexpected feature shape: {feat.shape}")

    def _concept_probs(self, x, projector, prototypes):
        """
        x: [1, N, C] or [B, N, C]
        returns p: [same batch, N, P]
        """
        x = projector(x)
        p = prototypes.protos.to(dtype=x.dtype, device=x.device)

        x = normalize_mean_std(x)
        p = normalize_mean_std(p)

        M = l2_dist(x, p)
        probs = F.softmax(-M / self.args.concept_sigma, dim=-1)

        return probs

    def _concept_loss(
        self,
        feat_s,
        feat_t,
        projector_s,
        projector_t,
        prototypes,
        mapping_probs,
        stage_key,
    ):
        tok_s_all = self._local_tokens(feat_s)  # [B, 196, Cs]
        tok_t_all = self._local_tokens(feat_t)  # [B, T, Ct]

        B, L, _ = tok_s_all.shape
        T = tok_t_all.shape[1]

        assert mapping_probs.shape[0] == L, (
            f"Stage {stage_key}: mapping has L={mapping_probs.shape[0]}, "
            f"but student has L={L}"
        )
        assert mapping_probs.shape[1] == T, (
            f"Stage {stage_key}: mapping has T={mapping_probs.shape[1]}, "
            f"but teacher has T={T}"
        )

        K = min(self.args.concept_k, B * L)

        idx = torch.randperm(B * L, device=tok_s_all.device)[:K]
        b_idx = idx // L
        l_idx = idx % L

        tok_s = tok_s_all[b_idx, l_idx].unsqueeze(0)  # [1, K, Cs]
        p_s = self._concept_probs(tok_s, projector_s, prototypes)  # [1, K, P]

        p_t_all = self._concept_probs(tok_t_all, projector_t, prototypes)  # [B, T, P]

        A = mapping_probs.to(device=p_t_all.device, dtype=p_t_all.dtype)  # [196, T]
        p_t_aligned_all = torch.einsum("lt,btp->blp", A, p_t_all)  # [B, 196, P]

        if getattr(self.args, "concept_debug_save", False):
            self._debug_save_targets_and_exit(
                p_t_aligned_all=p_t_aligned_all,
                p_s=p_s,
                A=A,
                b_idx=b_idx,
                l_idx=l_idx,
                out_dir=os.path.join(getattr(self.args, "concept_debug_dir", "debug"), f"stage{stage_key}"),
                num_examples=getattr(self.args, "concept_debug_examples", 2),
                top_teacher=getattr(self.args, "concept_debug_top_teacher", 196),
            )

        p_t = p_t_aligned_all[b_idx, l_idx].unsqueeze(0)  # [1, K, P]

        q_t = distributed_sinkhorn_probs(
            p_t.detach(),
            nmb_iters=self.args.concept_sinkhorn_iters,
            sharpen=self.args.concept_sigma / self.args.concept_sinkhorn_eps,
            world_size=self.args.world_size,
        )

        loss_match = -torch.sum(
            p_t.detach() * torch.log(p_s + 1e-6),
            dim=-1,
        ).mean()

        loss_t_bal = -torch.sum(
            q_t.detach() * torch.log(p_t + 1e-6),
            dim=-1,
        ).mean()

        return (loss_match + loss_t_bal) / 2

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_t, feats_t = self.teacher(image, requires_feat=True)

        logits_s, feats_s = self.student(image, requires_feat=True)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_s, label)

        loss_kd = self.args.kd_loss_weight * kd_loss(
            logits_s,
            logits_t,
            temperature=self.args.kd_temperature,
        )

        loss_concept = 0.0

        for stage, stage_key in zip(self.stages, self.stage_keys):
            idx_s, _ = self.student.stage_info(stage)
            idx_t, _ = self.teacher.stage_info(stage)

            mapping_probs = getattr(self, f"mapping_probs_{stage_key}")

            stage_loss = self._concept_loss(
                feats_s[idx_s],
                feats_t[idx_t],
                self.projector_s[stage_key],
                self.projector_t[stage_key],
                self.prototypes[stage_key],
                mapping_probs,
                stage_key,
            )

            loss_concept = loss_concept + stage_loss

        return logits_s, {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
            "loss_concept": self.args.concept_loss_weight * loss_concept,
        }

    @torch.no_grad()
    def _debug_save_targets_and_exit(
        self,
        p_t_aligned_all,
        p_s,
        A,
        b_idx,
        l_idx,
        out_dir="debug",
        num_examples=8,
        top_teacher=20,
    ):
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)

        p_t_aligned_all = p_t_aligned_all.detach().float().cpu()
        p_s = p_s.detach().float().cpu()
        A = A.detach().float().cpu()
        b_idx = b_idx.detach().cpu()
        l_idx = l_idx.detach().cpu()

        num_examples = min(num_examples, len(b_idx))

        for n in range(num_examples):
            b = int(b_idx[n])
            l = int(l_idx[n])

            teacher_target = p_t_aligned_all[b, l]  # [P]
            student_pred = p_s[0, n]  # [P]
            a_row = A[l]  # [T]

            sorted_a, sorted_teacher_idx = torch.sort(a_row, descending=True)

            plt.figure(figsize=(12, 4))
            plt.plot(teacher_target.numpy(), label="teacher target")
            plt.plot(student_pred.numpy(), label="student pred", alpha=0.7)
            plt.xlabel("concept id")
            plt.ylabel("probability")
            plt.title(f"sample {n}: image={b}, student_patch={l}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"sample_{n:02d}_concept_distribution.png"), dpi=150)
            plt.close()

            k = min(top_teacher, sorted_a.numel())
            plt.figure(figsize=(12, 4))
            plt.bar(range(k), sorted_a[:k].numpy())
            plt.xticks(
                range(k),
                [str(int(i)) for i in sorted_teacher_idx[:k]],
                rotation=45,
                ha="right",
            )
            plt.xlabel("teacher token index, sorted by A")
            plt.ylabel("A weight")
            plt.title(f"sample {n}: top teacher tokens for student_patch={l}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"sample_{n:02d}_A_sorted.png"), dpi=150)
            plt.close()

            T = a_row.numel()
            side = int(T ** 0.5)
            if side * side == T:
                plt.figure(figsize=(5, 5))
                plt.imshow(a_row.reshape(side, side).numpy())
                plt.colorbar()
                plt.title(f"A heatmap: student_patch={l}, teacher grid={side}x{side}")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"sample_{n:02d}_A_heatmap.png"), dpi=150)
                plt.close()

        print(f"[ConceptKD debug] Saved {num_examples} examples to: {out_dir}")
        # sys.exit(0)