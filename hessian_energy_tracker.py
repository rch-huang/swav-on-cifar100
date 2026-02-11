# hessian_energy_tracker_swav.py
# ============================================================
# SwAV-consistent Hessian Energy Tracker (within one task)
#
# - Uses EXACT SwAV loss formula via model.swav_loss_from_q(...)
# - Uses your training distributed_sinkhorn (injected callable)
# - Separately tracks theta-block and C-block
# - Saves: raw eigvals/proj/energy + histograms + sanity logs
#
# Two call sites:
#   tracker.start_task(model, probe_batches, epoch0)
#   tracker.after_epoch(epoch, model)
# ============================================================

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Callable, Tuple


# ---------------------------
# helpers
# ---------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_cpu_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _flatten_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    if len(params) == 0:
        return torch.empty(0)
    return torch.cat([p.detach().reshape(-1) for p in params])
def _flatten_params_cpu(params):
    """
    Flatten parameters directly on CPU to avoid large GPU allocations.
    """
    vecs = []
    for p in params:
        vecs.append(p.detach().cpu().reshape(-1))
    return torch.cat(vecs, dim=0)


def _split_named_params(model) -> Dict[str, List[torch.nn.Parameter]]:
    """
    theta: all params except those whose name contains 'prototypes'
    C    : all params whose name contains 'prototypes'

    NOTE:
    This matches typical SwAV code where prototype weights are named with 'prototypes'.
    If your code uses different naming, adjust the string predicate here.
    """
    theta, C = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "prototypes" in name:
            C.append(p)
        else:
            theta.append(p)
    return {"theta": theta, "C": C}


def _sanity_projection(delta: torch.Tensor, eigvals: torch.Tensor, proj: torch.Tensor, tag: str) -> Dict[str, float]:
    """
    proj = U^T delta, energy = proj^2
    """
    energy = proj.pow(2)
    delta_norm2 = float(delta.pow(2).sum().item())
    proj_energy = float(energy.sum().item())
    coverage = proj_energy / (delta_norm2 + 1e-12) if delta_norm2 > 0 else 1.0

    return {
        f"{tag}/delta_norm2": delta_norm2,
        f"{tag}/proj_energy": proj_energy,
        f"{tag}/coverage_topK": coverage,
        f"{tag}/lambda_max": float(eigvals.max().item()) if eigvals.numel() > 0 else 0.0,
        f"{tag}/lambda_min": float(eigvals.min().item()) if eigvals.numel() > 0 else 0.0,
        f"{tag}/num_negative_lambda": float((eigvals < 0).sum().item()) if eigvals.numel() > 0 else 0.0,
    }


def _build_histograms(eigvals: torch.Tensor, energy: torch.Tensor, bin_size: int) -> Dict[str, Any]:
    """
    eigvals, energy are assumed already ordered by descending eigvals
    """
    K = eigvals.numel()
    total = energy.sum() + 1e-12

    fixed_energy = []
    fixed_ratio = []
    for i in range(0, K, bin_size):
        e = energy[i:i + bin_size].sum()
        fixed_energy.append(float(e.item()))
        fixed_ratio.append(float((e / total).item()))

    cumulative = torch.cumsum(energy, dim=0) / total
    log_energy = torch.log10(energy + 1e-20)

    return {
        "fixed_energy": fixed_energy,
        "fixed_ratio": fixed_ratio,
        "cumulative": _to_cpu_np(cumulative).tolist(),
        "log_energy": _to_cpu_np(log_energy).tolist(),
        "eigvals": _to_cpu_np(eigvals).tolist(),
    }


def _dump_raw_npz(save_dir: str, prefix: str, eigvals: torch.Tensor, proj: torch.Tensor, energy: torch.Tensor) -> None:
    _ensure_dir(save_dir)
    np.savez(
        os.path.join(save_dir, f"{prefix}.npz"),
        eigvals=_to_cpu_np(eigvals),
        proj=_to_cpu_np(proj),
        energy=_to_cpu_np(energy),
    )


def _freeze_bn_stats(model) -> None:
    """
    Keep model in train() but freeze BN running stats by setting BN modules to eval().
    This is a standard compromise for 2nd-order analysis.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def _normalize_probe_batch(batch: Any) -> List[torch.Tensor]:
    """
    Normalize probe input to list-of-crops.
    Supports:
      - Tensor[B,C,H,W] (single crop)
      - list/tuple of tensors (multi-crop)
      - (inputs, y) where inputs is tensor or list/tuple
    """
    if isinstance(batch, (tuple, list)):
        # could be (inputs, y) or actual crops
        # heuristic: (inputs, y) usually has len==2 and second is NOT 4D image batch
        if len(batch) == 2:
            maybe_inputs, maybe_y = batch
            if torch.is_tensor(maybe_inputs) or isinstance(maybe_inputs, (list, tuple)):
                # treat as (inputs, y)
                inputs = maybe_inputs
            else:
                inputs = batch  # fallback
        else:
            inputs = batch
    else:
        inputs = batch

    if torch.is_tensor(inputs):
        assert inputs.dim() == 4, f"Probe input tensor must be 4D [B,C,H,W], got {inputs.shape}"
        return [inputs]
    elif isinstance(inputs, (list, tuple)):
        inputs = list(inputs)
        for x in inputs:
            assert torch.is_tensor(x) and x.dim() == 4, f"Each crop must be 4D tensor, got {type(x)} {getattr(x,'shape',None)}"
        return inputs
    else:
        raise TypeError(f"Unsupported probe batch type: {type(batch)}")


def _check_q_sanity(q: torch.Tensor, tag: str) -> Dict[str, float]:
    """
    Basic sanity on assignment matrix q: shape [bs,K], nonnegative, row-sums ~ 1.
    """
    stats = {}
    stats[f"{tag}/q_min"] = float(q.min().item())
    stats[f"{tag}/q_max"] = float(q.max().item())
    stats[f"{tag}/q_neg_frac"] = float((q < -1e-8).float().mean().item())
    rowsum = q.sum(dim=1)
    stats[f"{tag}/q_rowsum_mean"] = float(rowsum.mean().item())
    stats[f"{tag}/q_rowsum_std"] = float(rowsum.std(unbiased=False).item())
    stats[f"{tag}/q_rowsum_min"] = float(rowsum.min().item())
    stats[f"{tag}/q_rowsum_max"] = float(rowsum.max().item())
    stats[f"{tag}/q_nan"] = float(torch.isnan(q).any().item())
    stats[f"{tag}/q_inf"] = float(torch.isinf(q).any().item())
    return stats


# ---------------------------
# main tracker
# ---------------------------
allowed_blocks = {"C", "theta"}
class HessianEnergyTrackerSwAV:
    """
    SwAV-consistent Hessian tracker.

    Two call sites:
      - start_task(model, probe_batches, epoch0=0)
      - after_epoch(epoch, model)
    """

    def __init__(
        self,
        *,
        anchor_epochs: List[int],
        window1: int,
        window2: int,
        top_k_theta: int = 50,
        top_k_C: int = 500,
        bin_size: int = 50,
        save_root: str = "./hessian_energy_swav",
        device: str = "cuda",
        # swav config
        crops_for_assign: List[int],
        nmb_crops: int,
        temperature: float,
        clamp_min: Optional[float],
        total_epoch: int,
        # your sinkhorn
        sinkhorn_fn: Callable[..., Any],
        prototypes_head_index: int = 0,
        # lanczos config (correctness-focused)
        lanczos_m: Optional[int] = None,   # if None, use 2*top_k+20
        lanczos_reorth: bool = True,
        lanczos_seed: int = 0,
    ):
        self.anchor_epochs = set(anchor_epochs)
        self.window1 = int(window1)
        self.window2 = int(window2)
        self.top_k_theta = int(top_k_theta)
        self.top_k_C = int(top_k_C)
        self.bin_size = int(bin_size)
        self.save_root = save_root
        self.device = device
        self.rank = 0  # single-process default unless you set externally
        self.current_task: Optional[int] = None
        self._task_optimal: Dict[int, Dict[str, torch.Tensor]] = {}  # task -> block -> flat cpu tensor
        self._task_optimal_epoch: Dict[int, int] = {}
        self._task_optimal_basis: Dict[int, Dict[str, Any]] = {}  # task -> block -> {params,eigvals,eigvecs,top_k,epoch}
        self.window_steps1 = int(window1)  # interpret window as number of STEPS after anchor (in next epoch)
        self.window_steps2 = int(window2)

        self.crops_for_assign = list(crops_for_assign)
        self.nmb_crops = int(nmb_crops)
        self.temperature = float(temperature)
        self.clamp_min = clamp_min
        self.total_epoch = int(total_epoch)
        self.sinkhorn_fn = sinkhorn_fn
        self.prototypes_head_index = int(prototypes_head_index)

        self.lanczos_m = lanczos_m
        self.lanczos_reorth = bool(lanczos_reorth)
        self.lanczos_seed = int(lanczos_seed)

        self._probe_batches: Optional[List[Any]] = None
        # Optional: probe batches for the PREVIOUS task (used to compute Rayleigh denom on old-task loss)
        self._probe_batches_previous_task: Optional[List[Any]] = None
        self._anchors: Dict[int, Dict[str, Any]] = {}
        self._sanity_log: Dict[int, List[Dict[str, float]]] = {}
        self._epoch0 = 0
        self.plot_every = 1
        _ensure_dir(self.save_root)

    # -------------------------
    def _task_root(self, task: Optional[int] = None) -> str:
        """Return per-task root directory under save_root."""
        t = int(task) if task is not None else (int(self.current_task) if self.current_task is not None else -1)
        return os.path.join(self.save_root, f"task_{t}")

    # public API (1)
    # -------------------------
    # -------------------------
    # public API (1)
    # -------------------------
    def start_task(
        self,
        model,
        probe_batches: List[Any],
        probe_batches_previous_task: Optional[List[Any]] = None,
        epoch0: int = 0,
        task: Optional[int] = None,
    ) -> None:
        """
        Reset per-task state. probe_batches are stored and reused.

        Parameters
        ----------
        task: optional int
            Current task id (recommended). Used to save/load task-optimal snapshots for forgetting projections.
        """
        self._probe_batches = probe_batches
        self._probe_batches_previous_task = probe_batches_previous_task
        if task is not None:
            self.current_task = int(task)
        self._anchors.clear()
        self._sanity_log.clear()
        self._epoch0 = int(epoch0)
        _ensure_dir(self._task_root())

    # -------------------------
    # public API (2)
    # -------------------------
    def after_epoch(self, epoch: int, model, task: Optional[int] = None) -> None:
        """
        Epoch-level hook.

        - If current epoch is an anchor: compute & save Hessian eigens at this anchor.
        - If current epoch is the last epoch of a task: snapshot task-optimal parameters (for next task forgetting probes).
        - NOTE: delta projections are handled in after_step().
        """
        if task is not None:
            self.current_task = int(task)

        # 1) anchor registration at this epoch
        if epoch in self.anchor_epochs:
            self._register_anchor(epoch, model)
            # do not return: last epoch may also save task-optimal

        # 2) save task-optimal snapshot at last epoch
        if epoch == self.total_epoch - 1:
            self._register_task_optimal(epoch, model)

        return

    def after_step(self, task: int, epoch: int, step: int, model) -> None:
        """
        Step-level hook.

        For each anchor epoch 'a', we only process deltas for:
          - epoch == a + 1
          - step in [0, window_steps-1]

        At each valid step, we compute TWO deltas (same anchor Hessian basis):
          1) delta_anchor = w_now - w_anchor
          2) delta_prevopt = w_now - w_prev_task_optimal (if exists)

        Both deltas are projected onto the anchor Hessian top-eigens and saved.
        """
        self.current_task = int(task)

        # fast skip: no anchors registered yet
        if len(self._anchors) == 0:
            return

        for a in list(self._anchors.keys()):
            if epoch != a + 1:
                continue
            if step < 0 or step >= self.window_steps1:
                continue
             
            is_window1_end = (step == self.window_steps1 - 1)
            #is_plot_epoch = (epoch % self.plot_every == 0)

            if not is_window1_end: #and is_plot_epoch):
                continue

            self._process_step_delta(task=task, epoch=epoch, step=step, anchor_epoch=a, model=model,window_suffix="window1")
        for a in list(self._anchors.keys()):
            if epoch != a + 1:
                continue
            if step < 0 or step >= self.window_steps2:
                continue
             
            is_window2_end = (step == self.window_steps2 - 1)
            #is_plot_epoch = (epoch % self.plot_every == 0)

            if not is_window2_end: #and is_plot_epoch):
                continue

            self._process_step_delta(task=task, epoch=epoch, step=step, anchor_epoch=a, model=model,window_suffix="window2")
        return

    # =========================================================

 
    def get_latest_anchor_epoch(self) -> Optional[int]:
        return next(iter(self._anchors.keys())) if len(self._anchors) > 0 else None

    def get_prev_epoch_hessian(self, epoch: int, block: str) -> Optional[Dict[str, torch.Tensor]]:
        """Return previous-epoch Hessian basis (anchor at epoch-1), if cached in RAM."""
        if block not in allowed_blocks:
            return None
        a = self.get_latest_anchor_epoch()
        if a is None or int(epoch) != int(a) + 1:
            return None
        return self._anchors[a].get(block, None)

    def get_prev_task_optimal(self, task: int, block: str) -> Optional[Dict[str, torch.Tensor]]:
        """Return previous-task optimal basis (H_{t-1}(w*_{t-1})) if cached in RAM."""
        if block not in allowed_blocks:
            return None
        prev_t = int(task) - 1
        return self._task_optimal_basis.get(prev_t, {}).get(block, None)

    # -------------------------
    # Skip logging (jsonl)
    # -------------------------
    def log_skip_step(self, *, task: int, epoch: int, step: int, payload: Dict[str, Any]) -> None:
        log_dir = os.path.join(self._task_root(task), "skip_logs")
        _ensure_dir(log_dir)
        path = os.path.join(log_dir, "skip_steps.jsonl")
        record = {"task": int(task), "epoch": int(epoch), "step": int(step), **payload}
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_skip_epoch_summary(self, *, task: int, epoch: int, payload: Dict[str, Any]) -> None:
        log_dir = os.path.join(self._task_root(task), "skip_logs")
        _ensure_dir(log_dir)
        path = os.path.join(log_dir, "skip_epochs.jsonl")
        record = {"task": int(task), "epoch": int(epoch), **payload}
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # SwAV probe loss (EXACT formula via model.swav_loss_from_q)
        # =========================================================

    def _probe_loss_swav(
        self,
        model,
        *,
        probe_batches: Optional[List[Any]] = None,
        epoch_for_sinkhorn: int,
        require_grad: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Returns:
            loss: scalar Tensor (requires_grad=True)
            sanity: dict of float stats (q sanity, loss sanity)
            """
            if probe_batches is None:
                probe_batches = self._probe_batches
            assert probe_batches is not None, "Call start_task() first (or pass probe_batches)."

            total_loss = None
            sanity_accum: Dict[str, float] = {}
            count = 0

            for batch in probe_batches:
                inputs = _normalize_probe_batch(batch)  # list of crops
                nmb_crops_actual = len(inputs)

                # forward (keep consistent geometry; freeze BN stats for stability)
                _freeze_bn_stats(model)
                emb, out = model(inputs)

                # if MultiPrototypes returns list of logits, pick head
                if isinstance(out, list):
                    out = out[self.prototypes_head_index]

                assert out.dim() == 2, f"Expected logits [N,K], got {out.shape}"
                bs = inputs[0].shape[0]
                K = out.shape[1]

                # Ensure out has enough rows for nmb_crops_actual
                # In your current setting (single-crop probe), out.shape[0] == bs
                assert out.shape[0] >= bs * nmb_crops_actual, (
                    f"Logits rows {out.shape[0]} < bs*nmb_crops_actual={bs*nmb_crops_actual}. "
                    f"Inputs crops={nmb_crops_actual}, bs={bs}."
                )

                # crops_for_assign must be valid indices under current nmb_crops_actual
                crops_for_assign_actual = [c for c in self.crops_for_assign if c < nmb_crops_actual]
                if len(crops_for_assign_actual) == 0:
                    crops_for_assign_actual = [0]

                # compute q_assign for crops_for_assign_actual (no grad)
                q_assign: Dict[int, torch.Tensor] = {}
                with torch.no_grad():
                    out_ng = out.detach()
                    for crop_id in crops_for_assign_actual:
                        logits_crop = out_ng[bs * crop_id: bs * (crop_id + 1)]
                        assert logits_crop.shape == (bs, K), f"logits_crop shape mismatch: {logits_crop.shape} vs {(bs,K)}"

                        q_res = self.sinkhorn_fn(
                            logits_crop,
                            tracker=None,
                            clamp_min=self.clamp_min,
                            epoch=epoch_for_sinkhorn,
                            total_epoch=self.total_epoch,
                        )
                        Q = q_res[0] if isinstance(q_res, (tuple, list)) else q_res
                        Q = Q[-bs:].contiguous()
                        assert Q.shape == (bs, K), f"Q shape mismatch: {Q.shape} vs {(bs,K)}"
                        q_assign[crop_id] = Q

                # exact swav loss via model method
                loss = model.swav_loss_from_q(
                    output=out,
                    q_assign=q_assign,
                    bs=bs,
                    temperature=self.temperature,
                    nmb_crops=nmb_crops_actual,
                    crops_for_assign=crops_for_assign_actual,
                )

                # loss sanity
                if require_grad and (not loss.requires_grad):
                    raise RuntimeError("[ProbeLoss] loss.requires_grad=False (graph broken).")
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"[ProbeLoss] loss is NaN/Inf: {loss}")

                # accumulate sanity
                bstats = {
                    "probe/bs": float(bs),
                    "probe/nmb_crops": float(nmb_crops_actual),
                    "probe/K": float(K),
                    "probe/loss": float(loss.detach().item()),
                }
                # add q sanity for first crop_id (and mean over crops)
                for crop_id in crops_for_assign_actual:
                    q = q_assign[crop_id]
                    qstats = _check_q_sanity(q, tag=f"probe/crop{crop_id}")
                    bstats.update(qstats)
                    break

                # average sanity (running mean)
                for k, v in bstats.items():
                    sanity_accum[k] = sanity_accum.get(k, 0.0) + float(v)

                total_loss = loss if total_loss is None else (total_loss + loss)
                count += 1

            assert count > 0
            loss_avg = total_loss / count
            for k in list(sanity_accum.keys()):
                sanity_accum[k] /= count

            return loss_avg, sanity_accum

    # =========================================================
    # Task-optimal snapshot (for forgetting projections)
    # =========================================================
    def _register_task_optimal(self, epoch: int, model) -> None:
        """
        Save a flat CPU snapshot of parameters at (task, epoch). This is used as "previous-task optimal"
        reference when computing forgetting projections under a new task's anchor Hessian.

        By convention you call this at epoch == total_epoch-1.
        """
        if self.current_task is None:
            # If user didn't pass task, we cannot index the snapshot reliably.
            print("[HessianEnergyTrackerSwAV] WARNING: current_task is None; skip saving task-optimal snapshot.")
            return

        t = int(self.current_task)
        split = _split_named_params(model)

        save_dir = os.path.join(self._task_root(t), "task_optimal", f"epoch_{epoch}")
        _ensure_dir(save_dir)

        self._task_optimal[t] = {}
        self._task_optimal_epoch[t] = int(epoch)

        for block in ["theta", "C"]:
            if block not in split:
                continue
            params = split[block]
            if len(params) == 0:
                continue
            w = _flatten_params_cpu(params)  # CPU flat
            self._task_optimal[t][block] = w.detach().cpu()
            np.save(os.path.join(save_dir, f"{block}_params.npy"), _to_cpu_np(w))

            # Also compute and save top-k Hessian eigens at task-optimal (for next-task P^(t) projections)
            top_k = self.top_k_theta if block == "theta" else self.top_k_C
            eigvals, eigvecs, lanczos_sanity = self._top_hessian_eigens_lanczos(
                model=model,
                params=params,
                top_k=top_k,
                epoch_for_sinkhorn=epoch,
                tag=f"task_optimal_t{t}_e{epoch}/{block}",
               probe_batches=self._probe_batches,
            )
            with open(os.path.join(save_dir, f"{block}_eigvals.json"), "w") as f:
                json.dump({"eigvals": _to_cpu_np(eigvals).tolist()}, f, indent=2)
            torch.save(eigvecs.detach().cpu(), os.path.join(save_dir, f"{block}_eigvecs.pt"))
            torch.save(w.detach().cpu(), os.path.join(save_dir, f"{block}_w.pt"))
            with open(os.path.join(save_dir, f"{block}_lanczos_sanity.json"), "w") as f:
                json.dump(lanczos_sanity, f, indent=2)

            # cache in RAM (used by controller during next task)
            self._task_optimal_basis.setdefault(t, {})
            self._task_optimal_basis[t][block] = {
                "params": w.detach().cpu(),
                "eigvals": eigvals.detach().cpu(),
                "eigvecs": eigvecs.detach().cpu(),
                "top_k": top_k,
                "epoch": int(epoch),
            }

        if self.rank == 0:
            print(f"[HessianEnergyTrackerSwAV] Saved task-optimal snapshot: task={t}, epoch={epoch} -> {save_dir}")
    def _process_step_delta(self, *, task: int, epoch: int, step: int, anchor_epoch: int, model, window_suffix:str) -> None:
        """
        Minimal step-level logging for stacked top-K_big eig-projection heatmaps.

        Stores ONLY:
        - proj: U^T Δw   (length K_big)
        - eigvals: corresponding λ (length K_big)
        - delta_norm2 (optional debug)
        """
        device = self.device

        anchor = self._anchors[anchor_epoch]
        split_now = _split_named_params(model)

        prev_task = int(task) - 1
        prev_opt_params = self._task_optimal.get(prev_task, None)
        prev_opt_basis = self._task_optimal_basis.get(prev_task, None)

        records = []

        for block, ainfo in anchor.items():
            if block not in allowed_blocks:
                continue

            params_now = split_now.get(block, [])
            if len(params_now) == 0:
                continue

            # current params
            w_now = _flatten_params_cpu(params_now).to(device=device, dtype=torch.float32)

            # ========== (A) anchor basis ==========
            w_anchor = ainfo["params"].to(device=device, dtype=torch.float32)
            U = ainfo["eigvecs"].to(device=device, dtype=torch.float32)      # [D, K_big]
            lam = ainfo["eigvals"].to(device=device, dtype=torch.float32)   # [K_big]

            delta = (w_now - w_anchor).contiguous()
            proj = (U.T @ delta).contiguous()

            records.append({
                "task": int(task),
                "epoch": int(epoch),
                "step": int(step),
                "anchor_epoch": int(anchor_epoch),
                "block": block,
                "basis": "anchor",
                "window_suffix": window_suffix,
                "proj": proj.detach().cpu().numpy(),
                "eigvals": lam.detach().cpu().numpy(),
                "delta_norm2": float(delta.pow(2).sum().item()),
            })

            # ========== (B) previous-task optimal basis ==========
            if prev_opt_params is None or prev_opt_basis is None:
                continue
            if block not in prev_opt_params or block not in prev_opt_basis:
                continue

            w_star = prev_opt_params[block].to(device=device, dtype=torch.float32)
            U_p = prev_opt_basis[block]["eigvecs"].to(device=device, dtype=torch.float32)
            lam_p = prev_opt_basis[block]["eigvals"].to(device=device, dtype=torch.float32)

            delta_p = (w_now - w_star).contiguous()
            proj_p = (U_p.T @ delta_p).contiguous()

            records.append({
                "task": int(task),
                "epoch": int(epoch),
                "step": int(step),
                "anchor_epoch": int(anchor_epoch),
                "block": block,
                "basis": f"prev_task_{prev_task}",
                "window_suffix": window_suffix,
                "proj": proj_p.detach().cpu().numpy(),
                "eigvals": lam_p.detach().cpu().numpy(),
                "delta_norm2": float(delta_p.pow(2).sum().item()),
            })

        # ---- write to jsonl ----
        if records:
            save_dir = os.path.join(self._task_root(), "step_projections")
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, "proj.jsonl")

            with open(path, "a") as f:
                for r in records:
                    r["proj"] = r["proj"].tolist()
                    r["eigvals"] = r["eigvals"].tolist()
                    f.write(json.dumps(r) + "\n")

    # def _process_step_delta(self, *, task: int, epoch: int, step: int, anchor_epoch: int, model) -> None:
    #     """Step-level logging for stacked Hessian-energy heatmaps.

    #     For each valid anchor window end, we save two *independent* bases:

    #     (1) Anchor basis (current task):
    #         Δ_a = w_now - w_anchor
    #         proj_a[i] = u_i(anchor)^T Δ_a
    #         denom_a  = Δ_a^T H_anchor(w_anchor) Δ_a      (whole-Hessian Rayleigh, computed via HVP on CURRENT probe batches)

    #     (2) Previous-task optimal basis (old task), if exists:
    #         Δ_p = w_now - w_{t-1}^*
    #         proj_p[i] = u_i(prev-opt)^T Δ_p
    #         denom_p  = Δ_p^T H_prev(w_{t-1}^*) Δ_p      (whole-Hessian Rayleigh, computed via HVP on PREV-TASK probe batches)

    #     Each record stores:
    #         - proj (length K)
    #         - eigvals (length K; same order)
    #         - denom_rayleigh (scalar)
    #         - delta_norm2
    #     """
    #     device = self.device

    #     anchor = self._anchors[anchor_epoch]
    #     split_now = _split_named_params(model)

    #     prev_task = int(task) - 1
    #     prev_opt_params = self._task_optimal.get(prev_task, None)
    #     prev_opt_basis = self._task_optimal_basis.get(prev_task, None)

    #     # -------------------------
    #     # helper: Rayleigh Δ^T H Δ using current model params & a given probe_batches
    #     # -------------------------
        
    #     def _assign_flat_to_params_(params_list: List[torch.nn.Parameter], flat_vec: torch.Tensor) -> None:
    #         """In-place assign a flat vector to a parameter list (p.data), preserving shapes."""
    #         offset = 0
    #         for p in params_list:
    #             numel = p.numel()
    #             chunk = flat_vec[offset: offset + numel].view_as(p)
    #             p.data.copy_(chunk)
    #             offset += numel
    #         if offset != flat_vec.numel():
    #             raise RuntimeError(f"Flat size mismatch: used {offset}, got {flat_vec.numel()}")

    #     def _rayleigh_with_probe(
    #         delta_flat: torch.Tensor,
    #         params_list: List[torch.nn.Parameter],
    #         probe_batches: Optional[List[Any]],
    #         *,
    #         eval_flat_params: Optional[torch.Tensor] = None,
    #     ) -> float:
    #         """Compute Rayleigh quotient Δ^T H Δ for the SwAV probe loss defined on `probe_batches`.

    #         IMPORTANT:
    #         - If eval_flat_params is provided, the Hessian is evaluated at that parameter point
    #             (for this block only) by temporarily swapping `params_list` to `eval_flat_params`
    #             during the HVP computation, then restoring the current parameters.

    #         This keeps numerator/denominator on the SAME Hessian geometry for stacked heatmaps.
    #         """
    #         if probe_batches is None:
    #             return float("nan")

    #         # backup current block parameters if we need to evaluate at a different point
    #         backup_flat: Optional[torch.Tensor] = None
    #         if eval_flat_params is not None:
    #             with torch.no_grad():
    #                 backup_flat = torch.cat([p.data.detach().reshape(-1) for p in params_list]).clone()
    #                 _assign_flat_to_params_(params_list, eval_flat_params)

    #         try:
    #             loss, _ = self._probe_loss_swav(
    #                 model,
    #                 probe_batches=probe_batches,
    #                 epoch_for_sinkhorn=epoch,
    #                 require_grad=True,
    #             )
    #             grads = torch.autograd.grad(loss, params_list, create_graph=True, retain_graph=True)
    #             gflat = torch.cat([g.reshape(-1) for g in grads])
    #             hv = torch.autograd.grad(gflat @ delta_flat, params_list, retain_graph=False)
    #             hvflat = torch.cat([h.reshape(-1) for h in hv]).detach()
    #             return float((delta_flat * hvflat).sum().item())
    #         finally:
    #             if backup_flat is not None:
    #                 with torch.no_grad():
    #                     _assign_flat_to_params_(params_list, backup_flat)

    #     records: List[Dict[str, Any]] = []

    #     for block, ainfo in anchor.items():
    #         if block not in allowed_blocks:
    #             continue

    #         params_now = split_now.get(block, [])
    #         if len(params_now) == 0:
    #             continue

    #         # ---- current params (flatten on CPU then move to device) ----
    #         w_now = _flatten_params_cpu(params_now).to(device=device, dtype=torch.float32)

    #         # =========================================================
    #         # (A) Anchor basis (current task)
    #         # =========================================================
    #         w_anchor = ainfo["params"].to(device=device, dtype=torch.float32)
    #         U_a = ainfo["eigvecs"].to(device=device, dtype=torch.float32)  # [D, K]
    #         lam_a = ainfo["eigvals"].to(device=device, dtype=torch.float32)  # [K]

    #         delta_a = (w_now - w_anchor).contiguous()
    #         proj_a = (U_a.T @ delta_a).contiguous()
    #         delta_a_norm2 = float(delta_a.pow(2).sum().item())

    #         denom_a = _rayleigh_with_probe(delta_a, params_now, self._probe_batches, eval_flat_params=w_anchor)
    #         if False:
    #             numer_topk_a = float((lam_a * proj_a.pow(2)).sum().item())
    #             if denom_a > 0 and numer_topk_a / denom_a > 1.05:
    #                 print(
    #                     "[WARN][anchor]",
    #                     f"topK/denom={numer_topk_a/denom_a:.3f}",
    #                     "task", task, "epoch", epoch, "step", step, "block", block
    #                 )
    #                 assert numer_topk_a / denom_a <= 1.05, "Top-K energy exceeds total Rayleigh energy by >5% (sanity fail)."
    #         records.append({
    #             "task": int(task),
    #             "epoch": int(epoch),
    #             "step": int(step),
    #             "anchor_epoch": int(anchor_epoch),
    #             "block": block,
    #             "basis": "anchor",
    #             "eig_order": ainfo.get("eig_order", "desc"),
    #             "proj": proj_a.detach().cpu().numpy(),
    #             "eigvals": lam_a.detach().cpu().numpy(),
    #             "delta_norm2": float(delta_a_norm2),
    #             "denom_rayleigh": float(denom_a),
    #         })

    #         # =========================================================
    #         # (B) Previous-task optimal basis (old task) — requires prev params + prev eigbasis + prev probe batches
    #         # =========================================================
    #         if prev_opt_params is None or prev_opt_basis is None:
    #             continue
    #         if block not in prev_opt_params:
    #             continue
    #         if block not in prev_opt_basis:
    #             continue
    #         if self._probe_batches_previous_task is None:
    #             continue

    #         w_star = prev_opt_params[block].to(device=device, dtype=torch.float32)
    #         U_p = prev_opt_basis[block]["eigvecs"].to(device=device, dtype=torch.float32)
    #         lam_p = prev_opt_basis[block]["eigvals"].to(device=device, dtype=torch.float32)

    #         delta_p = (w_now - w_star).contiguous()
    #         proj_p = (U_p.T @ delta_p).contiguous()
    #         delta_p_norm2 = float(delta_p.pow(2).sum().item())

    #         denom_p = _rayleigh_with_probe(
    #             delta_p,
    #             params_now,
    #             self._probe_batches_previous_task,
    #             eval_flat_params=w_star,)

    #         records.append({
    #             "task": int(task),
    #             "epoch": int(epoch),
    #             "step": int(step),
    #             "anchor_epoch": int(anchor_epoch),
    #             "block": block,
    #             "basis": f"prev_task_{prev_task}",
    #             "eig_order": "desc",
    #             "proj": proj_p.detach().cpu().numpy(),
    #             "eigvals": lam_p.detach().cpu().numpy(),
    #             "delta_norm2": float(delta_p_norm2),
    #             "denom_rayleigh": float(denom_p),
    #             "prev_opt_epoch": int(prev_opt_basis[block].get("epoch", -1)),
    #         })

    #     # ---- append to jsonl buffer ----
    #     if records:
    #         save_dir = os.path.join(self._task_root(), "step_projections")
    #         _ensure_dir(save_dir)
    #         path = os.path.join(save_dir, "proj.jsonl")

    #         with open(path, "a") as f:
    #             for r in records:
    #                 r["proj"] = r["proj"].tolist()
    #                 r["eigvals"] = r["eigvals"].tolist()
    #                 f.write(json.dumps(r) + "\n")


    

    def _register_anchor(self, epoch: int, model) -> None:
        """
        Register Hessian anchor at epoch `epoch`.

        Minimal version:
        - Compute top-K Hessian eigens (for skip in next epoch)
        - Keep ONLY the latest anchor in RAM
        - Do NOT dump any plotting artifacts (w / eigvecs / eigvals)
        """

        print(f"[HessianEnergyTrackerSwAV] Register anchor epoch {epoch}")
        split = _split_named_params(model)

        anchor_data: Dict[str, Any] = {}

        for block in ["theta", "C"]:
            params = split[block]
            if len(params) == 0:
                continue
            print(f"  Processing block '{block}' with {len(params)} params...")   
            # NOTE: w_anchor only used to define delta in next epoch
            w_anchor = _flatten_params_cpu(params).to(self.device)
            top_k = self.top_k_theta if block == "theta" else self.top_k_C

            # Compute Hessian eigens (SwAV-native, unchanged)
            eigvals, eigvecs, lanczos_sanity = self._top_hessian_eigens_lanczos(
                model=model,
                params=params,
                top_k=top_k,
                epoch_for_sinkhorn=epoch,
                tag=f"anchor{epoch}/{block}",
                probe_batches=self._probe_batches,
            )

            # Keep only what is strictly needed for skip
            anchor_data[block] = {
                "params": w_anchor.detach(),   # stays on device
                "eigvals": eigvals.detach(),   # [K] stays on device (needed for Hessian-energy heatmap/controller)
                "eigvecs": eigvecs.detach(),   # stays on device
                "top_k": top_k,
                "eig_order": "desc",
            }

        # Keep ONLY the latest anchor in RAM
        self._anchors = {epoch: anchor_data}
        self._sanity_log = {epoch: []}

        # # Free unused CUDA memory
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    def _process_delta(self, epoch: int, anchor_epoch: int, model) -> None:
            anchor = self._anchors[anchor_epoch]
            split_now = _split_named_params(model)

            base_dir = os.path.join(self._task_root(), f"anchor_{anchor_epoch}")
            epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
            _ensure_dir(epoch_dir)

            row: Dict[str, float] = {"epoch": float(epoch)}
         
            for block, ainfo in anchor.items():
                if block not in allowed_blocks:
                    continue
                params_now = split_now[block]
                w_now = _flatten_params_cpu(params_now)
                delta = (w_now - ainfo["params"]).detach()

                eigvals = ainfo["eigvals"]
                U = ainfo["eigvecs"]  # [D, top_k]

                proj = (U.to(dtype=torch.float32).T @ delta.to(dtype=torch.float32)).contiguous()
                energy = proj.pow(2).contiguous()

                row.update(_sanity_projection(delta, eigvals, proj, tag=block))

                hist = _build_histograms(eigvals=eigvals, energy=energy, bin_size=self.bin_size)
                with open(os.path.join(epoch_dir, f"{block}_hist.json"), "w") as f:
                    json.dump(hist, f, indent=2)

                _dump_raw_npz(epoch_dir, f"{block}_raw", eigvals, proj, energy)

            self._sanity_log[anchor_epoch].append(row)
            with open(os.path.join(base_dir, "sanity.json"), "w") as f:
                json.dump(self._sanity_log[anchor_epoch], f, indent=2)

            print(f"[HessianEnergyTrackerSwAV] Done epoch {epoch} vs anchor {anchor_epoch}")

        # =========================================================
        # Correctness-focused top-k Hessian eigens via Lanczos
        # =========================================================

    def _top_hessian_eigens_lanczos(
    self,
    *,
    model,
    params: List[torch.nn.Parameter],
    top_k: int,
    epoch_for_sinkhorn: int,
    tag: str,
    probe_batches: Optional[List[Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Lanczos for symmetric operator H (exact Hessian via HVP).

        Returns:
        eigvals: [top_k] (descending)
        eigvecs: [D, top_k]
        sanity: dict (orthogonality, residuals, etc.)
        """
        import sys
        import time

        device = self.device
        D = sum(p.numel() for p in params)

        # lanczos dimension m
        m = self.lanczos_m
        if m is None:
            m = min(D, 2 * top_k + 20)
        m = int(m)
        if m < top_k + 2:
            m = top_k + 2

        # HVP for a flat vector v in R^D
        def hvp(v: torch.Tensor) -> torch.Tensor:
            v = v.to(device)

            loss, _ = self._probe_loss_swav(
                model,
                probe_batches=probe_batches,
                epoch_for_sinkhorn=epoch_for_sinkhorn,
                require_grad=True,
            )

            # ① first backward
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,
                retain_graph=True,
            )
            gflat = torch.cat([g.reshape(-1) for g in grads])

            # ② second backward
            hv = torch.autograd.grad(
                gflat @ v,
                params,
                retain_graph=False,
            )
            hvflat = torch.cat([h.reshape(-1) for h in hv]).detach()

            # explicit cleanup
            del loss, grads, gflat, hv
            return hvflat

        # initialize q0
        g = torch.Generator(device=device)
        g.manual_seed(self.lanczos_seed)
        q = torch.randn(D, device=device, dtype=torch.float32, generator=g)
        q = q / (q.norm() + 1e-12)

        Q = torch.zeros(D, m, device=device, dtype=torch.float32)
        alpha = torch.zeros(m, device=device, dtype=torch.float32)
        beta = torch.zeros(m, device=device, dtype=torch.float32)

        q_prev = torch.zeros_like(q)

        # >>> progress control (rank-safe)
        show_progress = True
        if hasattr(self, "rank"):
            show_progress = (self.rank == 0)
        progress_every = max(1, m // 50)  # ~50 updates max
        start_time = time.time()

        # Lanczos iterations
        k_eff = 0
        for k in range(m):
            Q[:, k] = q
            z = hvp(q)

            a = torch.dot(q, z)
            alpha[k] = a

            z = z - a * q - (beta[k - 1] * q_prev if k > 0 else 0.0)

            if self.lanczos_reorth and k > 0:
                coeff = Q[:, :k].T @ z
                z = z - Q[:, :k] @ coeff

            b = z.norm()
            beta[k] = b

            q_prev = q
            if b.item() < 1e-10:
                k_eff = k + 1
                break

            q = z / b
            k_eff = k + 1

            # >>> progress display (single-line, overwrite)
            if show_progress and (k % progress_every == 0 or k == m - 1):
                pct = 100.0 * (k + 1) / m
                elapsed = time.time() - start_time
                msg = (
                    f"[Lanczos {tag}] "
                    f"{k+1:4d}/{m} "
                    f"({pct:5.1f}%) "
                    f"elapsed {elapsed:6.1f}s"
                )
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()

        if show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # build tridiagonal T_k
        alpha_k = alpha[:k_eff]
        beta_k = beta[:k_eff]
        T = torch.diag(alpha_k)
        if k_eff > 1:
            off = beta_k[:k_eff - 1]
            T = T + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)

        # eigh of T
        evals_T, evecs_T = torch.linalg.eigh(T)
        idx = torch.argsort(evals_T, descending=True)
        evals_T = evals_T[idx]
        evecs_T = evecs_T[:, idx]

        # lift to R^D
        Qk = Q[:, :k_eff]
        U = Qk @ evecs_T

        # take top_k
        kk = min(top_k, k_eff)
        eigvals = evals_T[:kk].contiguous()
        eigvecs = U[:, :kk].contiguous()

        # -------- sanity checks ----------
        sanity: Dict[str, Any] = {}
        sanity[f"{tag}/D"] = int(D)
        sanity[f"{tag}/m"] = int(m)
        sanity[f"{tag}/k_eff"] = int(k_eff)
        sanity[f"{tag}/eig_top1"] = float(eigvals[0].item()) if kk > 0 else 0.0
        sanity[f"{tag}/eig_topk"] = float(eigvals[kk - 1].item()) if kk > 0 else 0.0

        # orthonormality
        UtU = eigvecs.T @ eigvecs
        I = torch.eye(kk, device=device, dtype=UtU.dtype)
        sanity[f"{tag}/orth_err_fro"] = float((UtU - I).norm().item()) if kk > 0 else 0.0

        # residual norms (kk extra HVPs)
        res = []
        for i in range(kk):
            ui = eigvecs[:, i]
            Hui = hvp(ui)
            ri = (Hui - eigvals[i] * ui).norm() / (Hui.norm() + 1e-12)
            res.append(float(ri.item()))
        sanity[f"{tag}/residual_rel"] = res

        sanity[f"{tag}/num_negative_lambda_topk"] = int((eigvals < 0).sum().item())

        return eigvals.detach(), eigvecs.detach(), sanity


    # =========================================================
    # Backward-compatibility aliases (older tracker revisions)
    # =========================================================
    def _register_anchor_and_hessian(self, epoch: int, model) -> None:
        """Alias of _register_anchor (older name)."""
        return self._register_anchor(epoch, model)

    def _register_anchor_epoch(self, epoch: int, model) -> None:
        """Alias of _register_anchor (older name)."""
        return self._register_anchor(epoch, model)

    def _process_forgetting_projection(self, *args, **kwargs):
        """Alias of _process_delta (older name)."""
        return self._process_delta(*args, **kwargs)