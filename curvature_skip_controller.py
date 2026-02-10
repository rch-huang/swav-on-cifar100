
# curvature_skip_controller_updated2.py
# ------------------------------------------------------------
# Curvature-aware step skipping controller (NO Hessian computation here).
#
# - Uses Hessian eigens computed by HessianEnergyTrackerSwAV (tracker).
# - Controller is decision-only: it holds a reference to tracker to read
#   (a) previous-epoch Hessian basis (for current-task curvature)
#   (b) previous-task optimal Hessian basis (for old-task curvature)
# - Controller does NOT save anything. Any logging is delegated to tracker.
# - Supports random baseline: replay skip counts from a previous skip log file.
# ------------------------------------------------------------

import json
import os
from typing import Any, Dict, Optional, Set, Tuple, List

import numpy as np
import torch


def _split_named_params(model) -> Dict[str, List[torch.nn.Parameter]]:
    theta, C = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "prototypes" in name:
            C.append(p)
        else:
            theta.append(p)
    return {"theta": theta, "C": C}


def _flatten_params_cpu(params: List[torch.nn.Parameter]) -> torch.Tensor:
    vecs = []
    for p in params:
        vecs.append(p.detach().cpu().reshape(-1))
    return torch.cat(vecs, dim=0) if len(vecs) > 0 else torch.empty(0)


class CurvatureSkipController:
    """Decision-only curvature skip controller.

    Two independent switches:
      - tau_curr: threshold for coverage on previous-epoch Hessian basis (current task).
      - tau_prev: threshold for coverage on previous-task optimal Hessian basis.

    tau == -1 disables that rule.

    Modes:
      - skip_mode="hessian": compute coverage from tracker-provided bases.
      - skip_mode="random": for each (task, epoch), randomly skip the same number of steps as recorded
        in a previous run's skip_steps.jsonl (or any jsonl containing 'skipped': true entries).
    """

    def __init__(
        self,
        *,
        skip_mode: str = "hessian",
        tau_curr_c: float = -1.0,
        tau_prev_c: float = -1.0,
        tau_curr_theta: float = -1.0,
        tau_prev_theta: float = -1.0,
        topk_theta: Optional[int] = None,
        topk_C: Optional[int] = None,
        m_anchor_theta: int = -1,
        m_anchor_C: int = -1,
        m_optimal_theta: int = -1,
        m_optimal_C: int = -1,
        use_theta: bool = True,
        use_C: bool = True,
        # random baseline
        replay_skip_log: Optional[str] = None,
        random_seed: int = 0,
        device: Optional[torch.device] = None,
    ):
        assert skip_mode in ("hessian", "random")
        self.topk_theta = topk_theta
        self.topk_C = topk_C
        self.skip_mode = skip_mode
        self.tau_curr_c = float(tau_curr_c)
        self.tau_prev_c = float(tau_prev_c)
        self.tau_curr_theta = float(tau_curr_theta)
        self.tau_prev_theta = float(tau_prev_theta)
        self.device = device
        self.use_theta = bool(use_theta)
        self.use_C = bool(use_C)
        self.m_anchor_theta = int(m_anchor_theta)
        self.m_anchor_C = int(m_anchor_C)
        self.m_optimal_theta = int(m_optimal_theta)
        self.m_optimal_C = int(m_optimal_C)

        self.replay_skip_log = replay_skip_log
        self.random_seed = int(random_seed)

        self._tracker = None
        self._task: Optional[int] = None
        self._epoch: Optional[int] = None
        self._enabled: bool = False

        # random baseline cache: (task, epoch) -> skip_count
        self._replay_counts: Dict[Tuple[int, int], int] = {}
        # random baseline sampled steps for current epoch
        self._random_skip_steps: Set[int] = set()

        if self.skip_mode == "random":
            if not self.replay_skip_log:
                raise ValueError("skip_mode='random' requires replay_skip_log path.")
            self._replay_counts = self._load_replay_counts(self.replay_skip_log)

    # -------------------------
    # lifecycle hooks
    # -------------------------
    def start_task(self, *, task: int, tracker) -> None:
        self._task = int(task)
        self._tracker = tracker

    def start_epoch(self, *, task: int, epoch: int, tracker, total_steps: int) -> None:
        self._task = int(task)
        self._epoch = int(epoch)
        self._tracker = tracker
        self._enabled = True

        # random baseline: sample steps now
        self._random_skip_steps = set()
        if self.skip_mode == "random":
            key = (self._task, self._epoch)
            k = int(self._replay_counts.get(key, 0))
            k = max(0, min(k, int(total_steps)))
            rng = np.random.RandomState(self.random_seed + 100000 * self._task + self._epoch)
            if k > 0:
                self._random_skip_steps = set(rng.choice(int(total_steps), size=k, replace=False).tolist())

        # reset epoch summary counters (tracked by tracker logging, but controller can pass counts)
        self._epoch_skip_count = 0
        self._epoch_total_steps = int(total_steps)
        self._epoch_skip_block_count = {"theta": 0, "C": 0}
    def end_epoch(self) -> None:
        # delegate epoch summary logging to tracker
        if self._tracker is not None and self._task is not None and self._epoch is not None:
            payload = {
                "skip_mode": self.skip_mode,
                "tau_curr_c": self.tau_curr_c,
                "tau_prev_c": self.tau_prev_c,
                "tau_curr_theta": self.tau_curr_theta,
                "tau_prev_theta": self.tau_prev_theta,
                "skip_count": int(getattr(self, "_epoch_skip_count", 0)),
                "skip_block_count": {b: int(getattr(self, "_epoch_skip_block_count", {}).get(b, 0)) for b in ("theta", "C")},
                "total_steps": int(getattr(self, "_epoch_total_steps", 0)),
            }
            if hasattr(self._tracker, "log_skip_epoch_summary"):
                self._tracker.log_skip_epoch_summary(task=self._task, epoch=self._epoch, payload=payload)

#     # -------------------------
#     # core decision
#     # -------------------------
#     def should_skip(
#     self,
#     *,
#     task: int,
#     epoch: int,
#     step: int,
#     model,
# ) -> Tuple[set, Dict[str, Any]]:
#         """
#         Block-wise skip controller.

#         Return:
#             skip_blocks: set[str] ⊆ {"theta", "C"}
#             stats: dict

#         Call AFTER backward(), BEFORE optimizer.step().
#         """
#         device = self.device
#         t, e, s = int(task), int(epoch), int(step)

#         stats: Dict[str, Any] = {
#             "task": t,
#             "epoch": e,
#             "step": s,
#             "skip_mode": self.skip_mode,
#         }

#         skip_blocks = set()

#         # ------------------------------------------------------------
#         # Random skip (block-agnostic baseline)
#         # ------------------------------------------------------------
#         if self.skip_mode == "random":
#             if s in self._random_skip_steps:
#                 skip_blocks = {"theta", "C"}
#                 stats["reason"] = "random"
#             else:
#                 stats["reason"] = "none"

#             stats["skip_blocks"] = sorted(skip_blocks)
#             return skip_blocks, stats

#         # ------------------------------------------------------------
#         # Hessian-based skip
#         # ------------------------------------------------------------
#         if self._tracker is None:
#             stats["reason"] = "no_tracker"
#             stats["skip_blocks"] = []
#             return set(), stats

#         split = _split_named_params(model)

#         for block in ("theta", "C"):
#             if block == "theta" and not self.use_theta:
#                 continue
#             if block == "C" and not self.use_C:
#                 continue

#             params = split.get(block, [])
#             if len(params) == 0:
#                 continue

#             # ---- choose controller top-k ----
#             if block == "theta":
#                 topk = self.topk_theta
#                 tau_curr = self.tau_curr_theta
#                 tau_prev = self.tau_prev_theta
#             else:  # block == "C"
#                 topk = self.topk_C
#                 tau_curr = self.tau_curr_c
#                 tau_prev = self.tau_prev_c

#             # ---- current parameters ----
#             w_now = _flatten_params_cpu(params).to(device=device, dtype=torch.float32)

#             block_triggered = False

#             # =========================================================
#             # Rule 1: previous-epoch Hessian (current task)
#             # =========================================================
#             if tau_curr >= 0:
#                 ainfo = self._tracker.get_prev_epoch_hessian(e, block)
#                 if ainfo is not None:
#                     w_ref = ainfo["params"].to(device=device, dtype=torch.float32)
#                     U = ainfo["eigvecs"].to(device=device, dtype=torch.float32)

#                     # ---- controller-specific top-k truncation ----
#                     if topk is not None and topk > 0:
#                         U = U[:, : min(topk, U.shape[1])]

#                     delta = w_now - w_ref
#                     delta_norm2 = float(delta.pow(2).sum().item())

#                     if delta_norm2 > 0:
#                         proj = (U.T @ delta).contiguous()
#                         cov_curr = float(proj.pow(2).sum().item()) / (delta_norm2 + 1e-12)
#                     else:
#                         cov_curr = 0.0

#                     stats[f"{block}/cov_curr"] = cov_curr
#                     stats[f"{block}/tau_curr"] = tau_curr
#                     stats[f"{block}/topk"] = U.shape[1]

#                     if cov_curr >= tau_curr:
#                         block_triggered = True
#                         stats[f"{block}/trigger_curr"] = True

#             # =========================================================
#             # Rule 2: previous-task optimal Hessian
#             # =========================================================
#             if tau_prev >= 0:
#                 pinfo = self._tracker.get_prev_task_optimal(t, block)
#                 if pinfo is not None:
#                     w_star = pinfo["params"].to(device=device, dtype=torch.float32)
#                     U2 = pinfo["eigvecs"].to(device=device, dtype=torch.float32)

#                     # ---- controller-specific top-k truncation ----
#                     if topk is not None and topk > 0:
#                         U2 = U2[:, : min(topk, U2.shape[1])]

#                     delta2 = w_now - w_star
#                     delta2_norm2 = float(delta2.pow(2).sum().item())

#                     if delta2_norm2 > 0:
#                         proj2 = (U2.T @ delta2).contiguous()
#                         cov_prev = float(proj2.pow(2).sum().item()) / (delta2_norm2 + 1e-12)
#                     else:
#                         cov_prev = 0.0

#                     stats[f"{block}/cov_prev"] = cov_prev
#                     stats[f"{block}/tau_prev"] = tau_prev
#                     stats[f"{block}/topk"] = U2.shape[1]

#                     if cov_prev >= tau_prev:
#                         block_triggered = True
#                         stats[f"{block}/trigger_prev"] = True

#             if block_triggered:
#                 skip_blocks.add(block)

#         if skip_blocks:
#             # step-level
#             self._epoch_skip_count += 1
#             # block-level
#             for b in skip_blocks:
#                 self._epoch_skip_block_count[b] += 1

#         stats["skip_blocks"] = sorted(skip_blocks)
#         stats["skipped"] = bool(skip_blocks)

#         return skip_blocks, stats
    def should_skip(
    self,
    *,
    task: int,
    epoch: int,
    step: int,
    model,
):
        device = self.device
        t, e, s = int(task), int(epoch), int(step)

        stats = {
            "task": t,
            "epoch": e,
            "step": s,
            "skip_mode": self.skip_mode,
            "reasons": [],
        }

        skip_blocks = set()

        # ------------------------------------------------------------
        # Random baseline
        # ------------------------------------------------------------
        if self.skip_mode == "random":
            if s in self._random_skip_steps:
                skip_blocks = {"theta", "C"}
                stats["reasons"].append("random")
            stats["skip_blocks"] = sorted(skip_blocks)
            stats["skipped"] = bool(skip_blocks)
            return skip_blocks, stats

        # ------------------------------------------------------------
        # Hessian-based skip
        # ------------------------------------------------------------
        if self._tracker is None:
            stats["skip_blocks"] = []
            stats["skipped"] = False
            return set(), stats

        split = _split_named_params(model)

        for block in ("theta", "C"):
            if block == "theta" and not self.use_theta:
                continue
            if block == "C" and not self.use_C:
                continue

            params = split.get(block, [])
            if len(params) == 0:
                continue

            if block == "theta":
                k_big = self.topk_theta
                m_anchor = self.m_anchor_theta
                m_optimal = self.m_optimal_theta
                tau_curr = self.tau_curr_theta
                tau_prev = self.tau_prev_theta
            else:
                k_big = self.topk_C
                m_anchor = self.m_anchor_C
                m_optimal = self.m_optimal_C
                tau_curr = self.tau_curr_c
                tau_prev = self.tau_prev_c

            w_now = _flatten_params_cpu(params).to(device=device, dtype=torch.float32)
            block_triggered = False

            # =========================================================
            # Rule A: previous-epoch (anchor) Hessian
            # =========================================================
            if tau_curr >= 0 and m_anchor > 0:
                ainfo = self._tracker.get_prev_epoch_hessian(e, block)
                if ainfo is not None:
                    U = ainfo["eigvecs"].to(device=device, dtype=torch.float32)
                    w_ref = ainfo["params"].to(device=device, dtype=torch.float32)

                    if k_big is not None:
                        U = U[:, :min(k_big, U.shape[1])]

                    delta = w_now - w_ref
                    proj = U.T @ delta
                    energy = proj.pow(2)

                    denom = energy.sum() + 1e-12
                    num = energy[:min(m_anchor, energy.numel())].sum()
                    ratio = float((num / denom).item())

                    stats[f"{block}/anchor_ratio"] = ratio
                    stats[f"{block}/anchor_tau"] = tau_curr
                    stats[f"{block}/anchor_m"] = m_anchor

                    if ratio >= tau_curr:
                        block_triggered = True
                        stats["reasons"].append(f"{block}_anchor")

            # =========================================================
            # Rule B: previous-task optimal Hessian
            # =========================================================
            if tau_prev >= 0 and m_optimal > 0:
                pinfo = self._tracker.get_prev_task_optimal(t, block)
                if pinfo is not None:
                    U = pinfo["eigvecs"].to(device=device, dtype=torch.float32)
                    w_star = pinfo["params"].to(device=device, dtype=torch.float32)

                    if k_big is not None:
                        U = U[:, :min(k_big, U.shape[1])]

                    delta = w_now - w_star
                    proj = U.T @ delta
                    energy = proj.pow(2)

                    denom = energy.sum() + 1e-12
                    num = energy[:min(m_optimal, energy.numel())].sum()
                    ratio = float((num / denom).item())

                    stats[f"{block}/optimal_ratio"] = ratio
                    stats[f"{block}/optimal_tau"] = tau_prev
                    stats[f"{block}/optimal_m"] = m_optimal

                    if ratio >= tau_prev:
                        block_triggered = True
                        stats["reasons"].append(f"{block}_optimal")

            if block_triggered:
                skip_blocks.add(block)
                self._epoch_skip_block_count[block] += 1

        if skip_blocks:
            self._epoch_skip_count += 1

        stats["skip_blocks"] = sorted(skip_blocks)
        stats["skipped"] = bool(skip_blocks)
        return skip_blocks, stats


    # -------------------------
    # random baseline helpers
    # -------------------------
    @staticmethod
    def _load_replay_counts(path: str) -> Dict[Tuple[int, int], int]:
        """Parse a jsonl skip log and count how many steps were skipped per (task, epoch)."""
        counts: Dict[Tuple[int, int], int] = {}
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not bool(rec.get("skipped", False)):
                    continue
                t = int(rec.get("task", -1))
                e = int(rec.get("epoch", -1))
                if t < 0 or e < 0:
                    continue
                key = (t, e)
                counts[key] = counts.get(key, 0) + 1
        return counts