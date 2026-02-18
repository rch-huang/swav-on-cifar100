
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

            #w_now = _flatten_params_cpu(params).to(device=device, dtype=torch.float32)
            # STEP-BASED: use current gradients instead of parameter difference
            grads = []
            for p in params:
                if p.grad is None:
                    continue
                grads.append(p.grad.detach().reshape(-1))
            if len(grads) == 0:
                continue
            delta = torch.cat(grads).to(device=device, dtype=torch.float32)
            block_triggered = False


            if epoch == 3 and (step == 0 or step == 10):
                extra = self.compute_ratio_diag_with_hvp(
                    model=model,
                    block=block,
                    task=t,
                    epoch=e,
                    step=s,
                    epoch_for_sinkhorn=e,
                    m_diag=m_anchor,        # 例如 250/500
                    max_i=500,
                    use_grad_delta=True,    # δ 用当前梯度（与你现在 controller 一致）
                )
                stats[f"{block}/ratio_diag"] = extra.get("ratio_diag", None)
                stats[f"{block}/denom_rayleigh"] = extra.get("denom_rayleigh", None)
                stats[f"{block}/delta_norm2_hvp"] = extra.get("delta_norm2", None)

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

                    #delta = w_now - w_ref
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

                    #delta = w_now - w_star
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

    def compute_ratio_diag_with_hvp(
        self,
        *,
        model,
        block: str,                 # "theta" or "C"
        task: int,
        epoch: int,
        step: int,
        epoch_for_sinkhorn: int,    # 用于 sinkhorn 的 epoch（通常就传当前 epoch）
        m_diag: int = 250,          # 你要的 m（比如 250/500）
        max_i: int = 500,           # 只算前 max_i 个 prototype 的 M_ii（避免 K 太大时爆炸）
        use_grad_delta: bool = True,# True: δ=当前梯度(flat); False: δ=w_now-w_anchor
        probe_batches=None,         # None 就用 tracker._probe_batches
        crop_id_for_diag: int = 0,  # 用哪个 crop 的 logits 做 e_i 的 Jacobian pullback
    ) -> dict:
        """
        在 loss.backward() 之后调用：
        - 用 anchor 参数 + probe_batch 计算 H_anchor
        - 用 HVP 得到 denom = δ^T H_anchor δ
        - 计算 (近似) M_ii = (J^T e_i)^T H_anchor (J^T e_i)
        - 用 g_i = mean_batch( d loss / d out[:,i] ) 构造 ratio_diag

        返回一个 dict，你可以直接 merge 到 should_skip 的 stats 里。
        """

        assert block in ("theta", "C")
        if self._tracker is None:
            return {"ok": False, "reason": "no_tracker"}

        device = self.device if self.device is not None else next(model.parameters()).device
        t, e, s = int(task), int(epoch), int(step)

        # -----------------------------
        # inner helpers (全部写在函数里)
        # -----------------------------
        def _split_named_params_local(_model):
            theta, C = [], []
            for name, p in _model.named_parameters():
                if not p.requires_grad:
                    continue
                if "prototypes" in name:
                    C.append(p)
                else:
                    theta.append(p)
            return {"theta": theta, "C": C}

        def _flatten_params(params):
            if len(params) == 0:
                return torch.empty(0, device=device)
            return torch.cat([p.detach().reshape(-1).to(device=device) for p in params], dim=0)

        def _flatten_grads(params):
            gs = []
            for p in params:
                if p.grad is None:
                    continue
                gs.append(p.grad.detach().reshape(-1))
            if len(gs) == 0:
                return None
            return torch.cat(gs, dim=0).to(device=device, dtype=torch.float32)

        def _assign_flat_to_params_(params, flat_vec):
            # flat_vec: 1D tensor on device
            offset = 0
            with torch.no_grad():
                for p in params:
                    n = p.numel()
                    p.copy_(flat_vec[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype))
                    offset += n
            assert offset == flat_vec.numel()

        def _normalize_probe_batch(batch):
            # 复刻 tracker 的 _normalize_probe_batch 行为（最小版）
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    maybe_inputs, _maybe_y = batch
                    inputs = maybe_inputs
                else:
                    inputs = batch
            else:
                inputs = batch

            if torch.is_tensor(inputs):
                return [inputs]
            return list(inputs)

        def _freeze_bn_stats(_model):
            _model.train()
            for m in _model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        def _forward_logits_and_q_assign(*, batch, require_grad: bool):
            """
            基于 tracker._probe_loss_swav 的逻辑：
            emb, out = model(inputs)
            out: [N,K] or list -> pick head
            q_assign: no_grad sinkhorn on selected crops
            """
            tr = self._tracker
            inputs = _normalize_probe_batch(batch)
            nmb_crops_actual = len(inputs)

            _freeze_bn_stats(model)
            emb, out = model(inputs)

            if isinstance(out, list):
                out = out[getattr(tr, "prototypes_head_index", 0)]

            assert out.dim() == 2, f"Expected logits [N,K], got {out.shape}"
            bs = inputs[0].shape[0]
            K = out.shape[1]

            assert out.shape[0] >= bs * nmb_crops_actual, (
                f"Logits rows {out.shape[0]} < bs*nmb_crops_actual={bs*nmb_crops_actual}."
            )

            crops_for_assign = getattr(tr, "crops_for_assign", [0])
            crops_for_assign_actual = [c for c in crops_for_assign if c < nmb_crops_actual]
            if len(crops_for_assign_actual) == 0:
                crops_for_assign_actual = [0]

            q_assign = {}
            with torch.no_grad():
                out_ng = out.detach()
                for cid in crops_for_assign_actual:
                    logits_crop = out_ng[bs * cid: bs * (cid + 1)]
                    q_res = tr.sinkhorn_fn(
                        logits_crop,
                        tracker=None,
                        clamp_min=getattr(tr, "clamp_min", 1e-6),
                        epoch=epoch_for_sinkhorn,
                        total_epoch=getattr(tr, "total_epoch", 1),
                    )
                    Q = q_res[0] if isinstance(q_res, (tuple, list)) else q_res
                    Q = Q[-bs:].contiguous()
                    q_assign[cid] = Q

            if not require_grad:
                out = out.detach()

            return out, q_assign, bs, K, nmb_crops_actual, crops_for_assign_actual

        def _swav_loss_from_out_q(*, out, q_assign, bs, nmb_crops_actual, crops_for_assign_actual):
            tr = self._tracker
            loss = model.swav_loss_from_q(
                output=out,
                q_assign=q_assign,
                bs=bs,
                temperature=getattr(tr, "temperature", 0.1),
                nmb_crops=nmb_crops_actual,
                crops_for_assign=crops_for_assign_actual,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"[compute_ratio_diag_with_hvp] loss is NaN/Inf: {loss}")
            return loss

        def _rayleigh_hvp_at_anchor(*, delta_flat, params_list, anchor_flat):
            """
            返回 delta^T H(anchor) delta  (H 是 probe loss 对 params_list 的 Hessian)
            完全走 HVP： grad(loss) -> gflat, 再 grad(gflat @ delta)
            """
            backup = _flatten_params(params_list)  # current params
            try:
                _assign_flat_to_params_(params_list, anchor_flat)

                # 重新建图：probe loss (requires_grad=True)
                loss_anchor = None
                count = 0
                for b in probe_batches_use:
                    out, q_assign, bs, K, nmb_crops_actual, crops_for_assign_actual = _forward_logits_and_q_assign(
                        batch=b,
                        require_grad=True,
                    )
                    loss_b = _swav_loss_from_out_q(
                        out=out,
                        q_assign=q_assign,
                        bs=bs,
                        nmb_crops_actual=nmb_crops_actual,
                        crops_for_assign_actual=crops_for_assign_actual,
                    )
                    loss_anchor = loss_b if loss_anchor is None else (loss_anchor + loss_b)
                    count += 1
                loss_anchor = loss_anchor / max(count, 1)

                grads = torch.autograd.grad(loss_anchor, params_list, create_graph=True, retain_graph=True)
                gflat = torch.cat([g.reshape(-1) for g in grads])

                hv = torch.autograd.grad(gflat @ delta_flat, params_list, retain_graph=False)
                hvflat = torch.cat([h.reshape(-1) for h in hv]).detach()

                return float((delta_flat * hvflat).sum().item())
            finally:
                _assign_flat_to_params_(params_list, backup)

        # -----------------------------
        # 1) get anchor flat params
        # -----------------------------
        ainfo = self._tracker.get_prev_epoch_hessian(int(epoch), block)
        if ainfo is None:
            return {"ok": False, "reason": f"no_anchor_basis_for_{block}"}

        anchor_flat = ainfo["params"].to(device=device, dtype=torch.float32)

        # -----------------------------
        # 2) choose params list for block and build delta
        # -----------------------------
        split = _split_named_params_local(model)
        params_list = split.get(block, [])
        if len(params_list) == 0:
            return {"ok": False, "reason": f"no_params_in_block_{block}"}

        if probe_batches is None:
            probe_batches_use = getattr(self._tracker, "_probe_batches", None)
        else:
            probe_batches_use = probe_batches
        if probe_batches_use is None or len(probe_batches_use) == 0:
            return {"ok": False, "reason": "no_probe_batches"}

        if use_grad_delta:
            delta_flat = _flatten_grads(params_list)
            if delta_flat is None:
                return {"ok": False, "reason": "no_grads_for_delta"}
            delta_flat = delta_flat.to(device=device, dtype=torch.float32).contiguous()
        else:
            w_now = _flatten_params(params_list).to(device=device, dtype=torch.float32)
            delta_flat = (w_now - anchor_flat).contiguous()

        # -----------------------------
        # 3) denom = delta^T H_anchor delta  (HVP, no eigs)
        # -----------------------------
        denom_rayleigh = _rayleigh_hvp_at_anchor(
            delta_flat=delta_flat,
            params_list=params_list,
            anchor_flat=anchor_flat,
        )

        # -----------------------------
        # 4) Phase A: compute g_vec and v_i_flat list at CURRENT params (NO anchor swap here)
        # -----------------------------
        batch0 = probe_batches_use[0]
        out0, q0, bs0, K0, nmb_crops0, crops_for_assign0 = _forward_logits_and_q_assign(batch=batch0, require_grad=True)
        loss0 = _swav_loss_from_out_q(
            out=out0,
            q_assign=q0,
            bs=bs0,
            nmb_crops_actual=nmb_crops0,
            crops_for_assign_actual=crops_for_assign0,
        )

        grad_out0 = torch.autograd.grad(loss0, out0, retain_graph=True)[0]

        out0_crop = out0[bs0 * crop_id_for_diag: bs0 * (crop_id_for_diag + 1)]
        grad0_crop = grad_out0[bs0 * crop_id_for_diag: bs0 * (crop_id_for_diag + 1)]

        g_vec = grad0_crop.mean(dim=0).detach().to(device=device, dtype=torch.float32)

        K_eff = int(min(K0, max_i))

        # ---- compute v_i_flat (T e_i) and store; DO NOT call anchor HVP in this phase ----
        v_list = [None] * K_eff
        for i in range(K_eff):
            scalar_i = out0_crop[:, i].mean()
            grads_i = torch.autograd.grad(
                scalar_i, params_list,
                retain_graph=True, create_graph=False, allow_unused=True
            )
            flat_parts = []
            for gi in grads_i:
                if gi is None:
                    continue
                flat_parts.append(gi.reshape(-1))
            if len(flat_parts) == 0:
                v_list[i] = None
            else:
                v_list[i] = torch.cat(flat_parts, dim=0).detach().to(device=device, dtype=torch.float32).contiguous()

        # IMPORTANT: free the current graph before any in-place param swap
        del loss0, grad_out0, grad0_crop, out0_crop, out0

        # -----------------------------
        # 5) Phase B: compute denom and M_ii under ANCHOR Hessian (anchor swap allowed here)
        # -----------------------------
        denom_rayleigh = _rayleigh_hvp_at_anchor(
            delta_flat=delta_flat,
            params_list=params_list,
            anchor_flat=anchor_flat,
        )

        M_diag = torch.zeros(K_eff, device=device, dtype=torch.float32)
        cum_weighted = 0.0
        debug_prototype_wise = []
        for i in range(K_eff):
            v_i_flat = v_list[i]
            if v_i_flat is None:
                M_diag[i] = 0.0
                continue
            M_ii = _rayleigh_hvp_at_anchor(
                delta_flat=v_i_flat,
                params_list=params_list,
                anchor_flat=anchor_flat,
            )
            M_diag[i] = float(M_ii)
            if True:
                g_i = float(g_vec[i].item())
                weighted_i = (g_i ** 2) * float(M_ii)

                cum_weighted += weighted_i

                denom_safe = float(denom_rayleigh) + 1e-12

                single_ratio = weighted_i / denom_safe
                cumulative_ratio = cum_weighted / denom_safe

                print(
                    f"[Mii DEBUG] "
                    f"i={i:4d} | "
                    f"M_ii={float(M_ii):.4e} | "
                    f"g_i={g_i:.4e} | "
                    f"M_ii*g_i^2={weighted_i:.4e} | "
                    f"single_ratio={single_ratio:.4e} | "
                    f"cumulative_ratio={cumulative_ratio:.4e}"
                )
                debug_prototype_wise.append({
                    "i": i,
                    "M_ii": float(M_ii),
                    "g_i": g_i,
                    "weighted_i": weighted_i,
                    "single_ratio": single_ratio,
                    "cumulative_ratio": cumulative_ratio,
                })

        # -----------------------------
        # 6) ratio_diag (full-diag, not top-m by index)
        # -----------------------------
        g2 = (g_vec[:K_eff] ** 2)
        weighted = g2 * M_diag
        denom_diag = float(weighted.sum().item())

        ratio_diag = denom_diag / (float(denom_rayleigh) + 1e-12)


        # pack stats
        return {
            "ok": True,
            "block": block,
            "task": t,
            "epoch": e,
            "step": s,
            "denom_rayleigh": float(denom_rayleigh),
            "delta_norm2": float(delta_flat.pow(2).sum().item()),
            "K": int(K0),
            "K_eff": int(K_eff),
            "m_diag": int(m_use),
            "ratio_diag": float(ratio_diag),
            "prototype_wise_debug": debug_prototype_wise,
            # debug payload (你要可视化/验证就打开；默认别全存，太大)
            # "debug": {
            #     "g_vec_head": g_vec[: min(10, K_eff)].detach().cpu().tolist(),
            #     "M_diag_head": M_diag[: min(10, K_eff)].detach().cpu().tolist(),
            #     "weighted_head": weighted[: min(10, K_eff)].detach().cpu().tolist(),
            # },
        }
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