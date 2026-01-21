# gym_state_context_logger.py
from __future__ import annotations
import os, csv, json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pathlib import Path

# ------------------------------
# Internal: run-length compressor
# ------------------------------
class _StateContextCompressor:
    """
    Compresses state and (true/observed) context streams into run-length segments
    and writes them into CSVs with rows:
      env_id, episode, marker, repetitions, return, vals_json
    where `marker == 'EP_END'` marks the end of an episode and stores the return.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        context_keys: List[str],
        *,
        state_mode: Optional[str] = "bins",   # "bins" | "exact" | None
        context_mode: Optional[str] = "exact",# "bins" | "exact" | None
        state_bins_per_dim: int = 12,
        context_bins: Optional[Union[int, Dict[str, int]]] = None,
        round_ndigits: int = 6,
        out_state_csv: Optional[str] = None,
        out_true_context_csv: Optional[str] = None,
        out_observed_context_csv: Optional[str] = None,
        env_id: int = 0,
        split_csv_per_env: bool = False,
        log_observed_contexts: Optional[List[str]] = None,
        overwrite_logs: bool = True,
    ):
        self.context_keys = list(context_keys or [])
        self.overwrite_logs = bool(overwrite_logs)
        self._obs_tracking_enabled = bool(out_observed_context_csv)

        if self._obs_tracking_enabled:
            self.obs_context_keys = (
                list(log_observed_contexts) if log_observed_contexts is not None
                else list(self.context_keys)
            )
        else:
            self.obs_context_keys = []

        self.state_mode = state_mode
        self.context_mode = context_mode
        self.state_bins = int(state_bins_per_dim)
        self.context_bins = context_bins
        self.round_ndigits = int(round_ndigits)
        self.env_id = int(env_id)
        self.split_csv_per_env = bool(split_csv_per_env)

        def _final_path(path: Optional[str]) -> Optional[str]:
            if not path:
                return None
            dirn = os.path.dirname(path)
            if dirn:
                os.makedirs(dirn, exist_ok=True)
            if self.split_csv_per_env:
                root, ext = os.path.splitext(path)
                if not ext:
                    ext = ".csv"
                path = f"{root}_env{self.env_id}{ext}"
            if os.path.exists(path):
                if self.overwrite_logs:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                # if not overwriting, keep file as is (assume header present)
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(
                        ["env_id", "episode", "marker", "repetitions", "return", "vals"]
                    )
            return path

        self.out_state    = _final_path(out_state_csv)
        self.out_true_ctx = _final_path(out_true_context_csv)
        self.out_obs_ctx  = _final_path(out_observed_context_csv) if self._obs_tracking_enabled else None

        self.log_states = bool(self.out_state)
        self.log_true   = bool(self.out_true_ctx)
        self.log_obs    = bool(self.out_obs_ctx)

        # --- state bin edges ---
        self._state_dim = None
        self._state_edges: Optional[List[np.ndarray]] = None
        if self.state_mode:
            if isinstance(obs_space, spaces.Dict) and "obs" in obs_space.spaces:
                sbox: spaces.Box = obs_space["obs"]
                low, high = sbox.low, sbox.high
                self._state_dim = int(np.prod(sbox.shape))
            elif isinstance(obs_space, spaces.Box):
                low, high = obs_space.low, obs_space.high
                self._state_dim = int(np.prod(obs_space.shape))
            else:
                raise RuntimeError("Unsupported observation_space (expected Box or Dict with 'obs').")

            low = np.asarray(low, dtype=np.float64).reshape(-1)[: self._state_dim]
            high = np.asarray(high, dtype=np.float64).reshape(-1)[: self._state_dim]
            if self.state_mode == "bins":
                same = np.isclose(high, low, rtol=0.0, atol=1e-12)
                if np.any(same):
                    mid = 0.5 * (low + high)
                    low[same]  = mid[same] - 1.0
                    high[same] = mid[same] + 1.0
                self._state_edges = [
                    np.linspace(low[i], high[i], self.state_bins + 1) for i in range(self._state_dim)
                ]

        # --- context bin edges ---
        self._ctx_edges: Dict[str, np.ndarray] = {}
        if self.context_mode == "bins" and isinstance(obs_space, spaces.Dict) and "context" in obs_space.spaces:
            ctx_space: spaces.Dict = obs_space["context"]
            all_ctx_keys = list(dict.fromkeys(self.context_keys + (self.obs_context_keys if self.log_obs else [])))
            if isinstance(self.context_bins, int):
                bins_map = {k: self.context_bins for k in all_ctx_keys}
            elif isinstance(self.context_bins, dict):
                bins_map = {k: int(self.context_bins.get(k, 10)) for k in all_ctx_keys}
            else:
                bins_map = {k: 10 for k in all_ctx_keys}
            for k in all_ctx_keys:
                sp = ctx_space.spaces.get(k, None)
                if sp is None:
                    continue
                lo = float(np.asarray(sp.low).reshape(-1)[0])
                hi = float(np.asarray(sp.high).reshape(-1)[0])
                if abs(hi - lo) < 1e-8:
                    mid = 0.5 * (lo + hi)
                    lo, hi = mid - 1.0, mid + 1.0
                self._ctx_edges[k] = np.linspace(lo, hi, bins_map[k] + 1)

        # episode buffers
        self._ep = 0
        self._ret = 0.0
        self._state_curr = None
        self._state_series: List[Dict[str, Any]] = []
        self._true_ctx_curr = None
        self._true_ctx_series: List[Dict[str, Any]] = []
        self._obs_ctx_curr = None if not self.log_obs else None
        self._obs_ctx_series: List[Dict[str, Any]] = [] if self.log_obs else []

    # ---- encoding helpers ----
    def _extract_state_vec(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and "obs" in obs:
            v = np.asarray(obs["obs"], dtype=np.float64).reshape(-1)
        else:
            v = np.asarray(obs, dtype=np.float64).reshape(-1)
        return v[: self._state_dim] if self._state_dim is not None else v

    def _state_vals(self, obs: Any):
        if not self.state_mode:
            return None
        s = self._extract_state_vec(obs)
        if self.state_mode == "exact":
            return tuple(float(x) for x in np.round(s, self.round_ndigits))
        idxs = []
        assert self._state_edges is not None
        for d, edges in enumerate(self._state_edges):
            b = int(np.digitize(float(s[d]), edges) - 1)
            b = max(0, min(b, len(edges) - 2))
            idxs.append(b)
        return tuple(int(x) for x in idxs)

    def _ctx_vals(self, ctx_or_obs: Any, keys: List[str]):
        if not self.context_mode or not keys:
            return None
        if isinstance(ctx_or_obs, dict) and "context" in ctx_or_obs:
            ctx = ctx_or_obs["context"]
        else:
            ctx = ctx_or_obs
        row = []
        for k in keys:
            if ctx is None or k not in ctx or ctx[k] is None:
                return None
            v = float(np.asarray(ctx[k]).reshape(-1)[0])
            if self.context_mode == "exact":
                row.append(float(np.round(v, self.round_ndigits)))
            else:
                edges = self._ctx_edges.get(k, None)
                if edges is None:
                    edges = np.linspace(v - 1.0, v + 1.0, 11)
                    self._ctx_edges[k] = edges
                b = int(np.digitize(v, edges) - 1)
                b = max(0, min(b, len(edges) - 2))
                row.append(float(b))
        return tuple(row)

    # ---- public episode interface ----
    def begin_episode(self, episode_id: int, obs_init: Any, obs_ctx_init: Optional[Dict[str, Any]], true_ctx_init: Optional[Dict[str, Any]]):
        self._ep = int(episode_id)
        self._ret = 0.0
        self._state_series.clear()
        self._true_ctx_series.clear()
        if self.log_obs:
            self._obs_ctx_series.clear()

        s0 = self._state_vals(obs_init)
        t0 = self._ctx_vals(true_ctx_init, self.context_keys)
        o0 = self._ctx_vals(obs_ctx_init, self.obs_context_keys) if self.log_obs else None

        self._state_curr    = {"vals": s0, "rep": 0} if s0 is not None else None
        self._true_ctx_curr = {"vals": t0, "rep": 0} if t0 is not None else None
        self._obs_ctx_curr  = ({"vals": o0, "rep": 0} if (self.log_obs and o0 is not None) else None)

    def observe_step(self, obs_t: Any, obs_ctx_t: Optional[Dict[str, Any]], true_ctx_t: Optional[Dict[str, Any]], reward: float = 0.0):
        self._ret += float(reward)
        s_now = self._state_vals(obs_t)
        t_now = self._ctx_vals(true_ctx_t, self.context_keys)
        o_now = self._ctx_vals(obs_ctx_t, self.obs_context_keys) if self.log_obs else None

        if self._state_curr is not None and s_now is not None:
            if s_now == self._state_curr["vals"]:
                self._state_curr["rep"] += 1
            else:
                self._state_series.append({"vals": self._state_curr["vals"], "rep": self._state_curr["rep"]})
                self._state_curr = {"vals": s_now, "rep": 0}

        if self._true_ctx_curr is not None and t_now is not None:
            if t_now == self._true_ctx_curr["vals"]:
                self._true_ctx_curr["rep"] += 1
            else:
                self._true_ctx_series.append({"vals": self._true_ctx_curr["vals"], "rep": self._true_ctx_curr["rep"]})
                self._true_ctx_curr = {"vals": t_now, "rep": 0}

        if self.log_obs and self._obs_ctx_curr is not None and o_now is not None:
            if o_now == self._obs_ctx_curr["vals"]:
                self._obs_ctx_curr["rep"] += 1
            else:
                self._obs_ctx_series.append({"vals": self._obs_ctx_curr["vals"], "rep": self._obs_ctx_curr["rep"]})
                self._obs_ctx_curr = {"vals": o_now, "rep": 0}

    def end_episode(self):
        if self._state_curr is not None:
            self._state_series.append({"vals": self._state_curr["vals"], "rep": self._state_curr["rep"]})
        if self._true_ctx_curr is not None:
            self._true_ctx_series.append({"vals": self._true_ctx_curr["vals"], "rep": self._true_ctx_curr["rep"]})
        if self.log_obs and self._obs_ctx_curr is not None:
            self._obs_ctx_series.append({"vals": self._obs_ctx_curr["vals"], "rep": self._obs_ctx_curr["rep"]})

        # states
        if self.out_state and self._state_series:
            with open(self.out_state, "a", newline="") as f:
                w = csv.writer(f)
                for seg in self._state_series:
                    w.writerow([self.env_id, self._ep, "", seg["rep"], "", json.dumps(list(seg["vals"]))])
                w.writerow([self.env_id, self._ep, "EP_END", "", float(self._ret), ""])

        # true contexts
        if self.out_true_ctx and self._true_ctx_series:
            with open(self.out_true_ctx, "a", newline="") as f:
                w = csv.writer(f)
                for seg in self._true_ctx_series:
                    w.writerow([self.env_id, self._ep, "", seg["rep"], "", json.dumps(list(seg["vals"]))])
                w.writerow([self.env_id, self._ep, "EP_END", "", float(self._ret), ""])

        # observed contexts
        if self.out_obs_ctx and self._obs_ctx_series:
            with open(self.out_obs_ctx, "a", newline="") as f:
                w = csv.writer(f)
                for seg in self._obs_ctx_series:
                    w.writerow([self.env_id, self._ep, "", seg["rep"], "", json.dumps(list(seg["vals"]))])
                w.writerow([self.env_id, self._ep, "EP_END", "", float(self._ret), ""])


# --------------------------------
# Public: Gym wrapper (drop-in)
# --------------------------------
ContextGetter = Callable[[gym.Env, Any, Dict[str, Any]], Optional[Dict[str, Any]]]

class GymStateContextLogger(gym.Wrapper):
    """
    Drop-in Gymnasium wrapper that logs (compressed) state and contexts.

    Context sources (priority order for TRUE context):
      1) context_getter(env.unwrapped, obs, info) -> dict or None
      2) info[true_context_info_key]               -> dict
      3) getattr(env.unwrapped, true_context_attr) -> dict

    Observed context source:
      - If observation is a dict with key "context", we use that.
      - Else, obs_context_getter(env.unwrapped, obs, info) if provided.

    The wrapper preserves reset/step/seed/spec/action_space/observation_space and forwards everything else.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        # --- context configuration ---
        true_context_keys: List[str],
        observation_mode: Optional[str] = None,
        true_context_info_key: Optional[str] = None,
        true_context_attr: Optional[str] = None,
        context_getter: Optional[ContextGetter] = None,
        obs_context_getter: Optional[ContextGetter] = None,

        # --- encoding / logging config (same semantics as your original) ---
        state_mode: Optional[str] = "bins",
        context_mode: Optional[str] = "exact",
        state_bins_per_dim: int = 12,
        context_bins: Optional[Union[int, Dict[str, int]]] = None,
        round_ndigits: int = 6,
        out_state_csv: Optional[str] = None,
        out_true_context_csv: Optional[str] = None,
        out_observed_context_csv: Optional[str] = None,
        env_id: int = 0,
        split_csv_per_env: bool = False,
        log_observed_contexts: Optional[List[str]] = None,
        overwrite_logs: bool = True,
        verbose: bool = True
    ):
        super().__init__(env)

        self._obs_mode = observation_mode # "initial" | "live" | None
        # Only create the observed-context CSV in "initial" mode; otherwise skip it.
        obs_csv_effective = out_observed_context_csv if observation_mode == "initial" else None

        self._context_getter = context_getter
        self._obs_context_getter = obs_context_getter
        self._true_ctx_info_key = true_context_info_key
        self._true_ctx_attr = true_context_attr

        if verbose:
            print(
                f"\n[sc_tracker]: env_id={env_id}  "
                f"state_csv={out_state_csv}  true_ctx_csv={out_true_context_csv}  "
                f"obs_ctx_csv={obs_csv_effective}  obs_mode={self._obs_mode}\n"
            )

        self._compressor = _StateContextCompressor(
            obs_space=env.observation_space,
            context_keys=list(true_context_keys or []),
            state_mode=state_mode,
            context_mode=context_mode,
            state_bins_per_dim=state_bins_per_dim,
            context_bins=context_bins,
            round_ndigits=round_ndigits,
            out_state_csv=out_state_csv,
            out_true_context_csv=out_true_context_csv,      # always keep true-context CSV
            out_observed_context_csv=obs_csv_effective,     # only if "initial"
            env_id=env_id,
            split_csv_per_env=split_csv_per_env,
            log_observed_contexts=log_observed_contexts,
            overwrite_logs=overwrite_logs,
        )
        self._episode_counter = -1  # will increment on first reset

    # ----------------- helpers to fetch contexts -----------------
    def _get_true_context(self, obs: Any, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 1) custom callable
        if self._context_getter is not None:
            try:
                out = self._context_getter(self.env.unwrapped, obs, info)
                if out is not None:
                    return out
            except Exception:
                pass
        # 2) info key
        if self._true_ctx_info_key and self._true_ctx_info_key in info:
            val = info[self._true_ctx_info_key]
            if isinstance(val, dict):
                return val
        # 3) env attribute
        if self._true_ctx_attr and hasattr(self.env.unwrapped, self._true_ctx_attr):
            val = getattr(self.env.unwrapped, self._true_ctx_attr)
            if isinstance(val, dict):
                return val
        return None

    def _get_observed_context(self, obs: Any, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Preferred: obs["context"] if present
        if isinstance(obs, dict) and "context" in obs and isinstance(obs["context"], dict):
            return obs["context"]
        # Fallback: custom observed-context getter
        if self._obs_context_getter is not None:
            try:
                out = self._obs_context_getter(self.env.unwrapped, obs, info)
                if out is not None:
                    return out
            except Exception:
                pass
        return None

    # ----------------- Gym API passthrough + logging -----------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        # call parent wrapper chain
        obs, info = super().reset(seed=seed, options=options)

        # (optional) if you want to auto-close a half-finished episode on manual reset:
        # if self._state_series or self._true_ctx_series or self._obs_ctx_series:
        #     self._compressor.end_episode()

        self._episode_counter += 1
        true_ctx = self._get_true_context(obs, info)
        obs_ctx  = self._get_observed_context(obs, info)
        self._compressor.begin_episode(
            self._episode_counter,
            obs_init=obs,
            obs_ctx_init=obs_ctx,
            true_ctx_init=true_ctx
        )
        return obs, info

    def step(self, action: Any):
        # call parent wrapper chain
        obs, reward, terminated, truncated, info = super().step(action)

        true_ctx = self._get_true_context(obs, info)
        obs_ctx  = self._get_observed_context(obs, info)
        self._compressor.observe_step(
            obs_t=obs,
            obs_ctx_t=obs_ctx,
            true_ctx_t=true_ctx,
            reward=float(reward)
        )
        if terminated or truncated:
            self._compressor.end_episode()
        return obs, reward, terminated, truncated, info

    # Keep legacy seed() for older code paths (Gymnasium prefers reset(seed=...))
    def seed(self, seed: Optional[int] = None) -> Any:
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)  # type: ignore[attr-defined]
        return None

    # Attribute passthrough is already handled by gym.Wrapper, but keeping these for clarity:
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def spec(self):
        return getattr(self.env, "spec", None)
# ----------------- end of GymStateContextLogger -----------------
