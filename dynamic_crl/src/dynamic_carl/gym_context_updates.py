# === DROP-IN REPLACEMENTS ===
# Keeps the same public API:
# - you still call: updater = make_sinusoidal(...), etc.
# - updater(env, step, ctx, verbose=False)
# - updater.reset(env, episode_id, ctx=None)
# - updater.config (dict)
# IMPORTANT: create a FRESH updater instance per env (don’t reuse across envs).

# NOTE: THIS CODE CURRENTLY ONLY WORKS WITH THE CARTPOLE ENV AND ITS EASY TO ACCESS ATTRIBUTES
# IN THE FUTURE IT SHALLE BE GENERALIZED TO WORK WITH ATTR GETTERS/SETTERS OR ENV CONTEXTS

import os
import numpy as np

def _episode_seed(base_seed, episode_id, worker_id=0):
    """
    Deterministic per-episode seed.
    - base_seed: global/base seed (required for reproducibility)
    - episode_id: per-env episode counter (int)
    - worker_id: stable env index (e.g., vec env rank), default 0
    - salt: optional extra disambiguator (I removed it for simplicity)
    Returns a 64-bit int usable with numpy.default_rng.
    """
    bs = 0 if base_seed is None else int(base_seed)
    ss = np.random.SeedSequence([bs, int(episode_id), int(worker_id)])
    # uint32 is fine for RNG; if you prefer 64-bit, use dtype=np.uint64 and mask as needed
    return int(ss.generate_state(1, dtype=np.uint32)[0])

# -----------------------------
# Identity updater (no change)
# -----------------------------
class _Identity:
    """
    Identity updater: leaves the attribute unchanged.
    Useful as a baseline/no-op replacement that still
    conforms to the same API.
    """
    def __init__(
            self, attribute,
            min_val, max_val,
            seed): # seed arg for API consistency; unused
        self.attribute = attribute
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.config = {"type": "identity", "attribute": attribute}

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        # No-op; but ensure env attribute is consistent
        val = getattr(env.unwrapped, self.attribute, None)
        if val is not None:
            setattr(env.unwrapped, self.attribute, float(val))

    def __call__(self, env, step, ctx, verbose=False):
        val = float(getattr(env.unwrapped, self.attribute))
        if verbose:
            print(f"[id] {self.attribute} step {step}: {val:.6g} (unchanged)")
        return val

def make_identity(*args, **kwargs):
    return _Identity(*args, **kwargs)

# -----------------------------
# Continuous incrementer
# -----------------------------
class _ContinuousIncrementer:
    def __init__(self,
                 attribute,
                 delta=0.01,
                 min_val=0.0,
                 max_val=1.0,
                 direction="positive",
                 direction_prob=0.5,
                 *,
                 edge_mode="clip",            # "clip" | "reflect"
                 episode_direction=None,      # "up" | "down" | "random" | None
                 noise_std=0.0,
                 noise_std_frac_of_delta=None,
                 follow_predefined_prob=None,
                 seed=None):
        assert direction in {"positive", "negative", "both"}
        assert edge_mode in {"clip", "reflect"}
        if episode_direction is not None:
            assert episode_direction in {"up", "down", "random"}

        self.follow_predefined_prob = (None if follow_predefined_prob is None
                                   else float(follow_predefined_prob))
        self.attribute = attribute
        self.delta = float(delta)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

        self.direction = direction
        self.direction_prob = float(direction_prob)

        self.edge_mode = edge_mode
        self.episode_direction = episode_direction
        self.noise_std = float(noise_std)
        self.noise_std_frac_of_delta = (None if noise_std_frac_of_delta is None
                                        else float(noise_std_frac_of_delta))
        self.seed = seed

        self.rng = None
        self._sgn = +1.0
        self._noise_std_eff = abs(self.noise_std)

        self.config = {
            "type": "continuous_incrementer",
            "attribute": attribute,
            "delta": delta,
            "min_val": min_val, "max_val": max_val,
            "direction": direction, "direction_prob": direction_prob,
            "edge_mode": self.edge_mode,
            "episode_direction": self.episode_direction,
            "noise_std": self.noise_std,
            "noise_std_frac_of_delta": self.noise_std_frac_of_delta,
            "seed": seed,
            "follow_predefined_prob": self.follow_predefined_prob,
        }

    def _choose_initial_sign(self):
        if self.episode_direction == "up": return +1.0
        if self.episode_direction == "down": return -1.0
        if self.episode_direction == "random":
            return +1.0 if self.rng.random() >= 0.5 else -1.0
        if self.direction == "positive": return +1.0
        if self.direction == "negative": return -1.0
        return +1.0 if self.rng.random() < self.direction_prob else -1.0

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        s = _episode_seed(self.seed, episode_id, worker_id=worker_id)
        self.rng = np.random.default_rng(s)
        self._sgn = self._choose_initial_sign()
        self._noise_std_eff = (abs(self.noise_std_frac_of_delta * self.delta)
                               if self.noise_std_frac_of_delta is not None
                               else abs(self.noise_std))

    def __call__(self, env, step, ctx, verbose=False):
        if self.rng is None:
            self.reset(env, 0, ctx)

        cur = float(getattr(env.unwrapped, self.attribute))

        # pick the persistent "predefined" sign
        sgn_persist = self._sgn if (self.edge_mode == "reflect" or self.episode_direction is not None or self.direction != "both") \
                    else (+1.0 if self.rng.random() < self.direction_prob else -1.0)

        # apply per-step noise relative to the predefined sign:
        if self.follow_predefined_prob is not None:
            # with prob p, go with predefined; with 1-p, go opposite
            sgn = sgn_persist if (self.rng.random() < self.follow_predefined_prob) else -sgn_persist
        else:
            # old semantics
            if self.edge_mode == "reflect" and self.direction == "both":
                sgn = -sgn_persist if (self.rng.random() < self.direction_prob) else sgn_persist
            else:
                sgn = sgn_persist

        change = sgn * self.delta
        if self._noise_std_eff > 0.0:
            change += float(self.rng.normal(0.0, self._noise_std_eff))

        proposed = cur + change

        if self.edge_mode == "reflect":
            if proposed < self.min_val:
                over = self.min_val - proposed
                new = self.min_val + over
                self._sgn *= -1.0  # persistent flip on bounce
            elif proposed > self.max_val:
                over = proposed - self.max_val
                new = self.max_val - over
                self._sgn *= -1.0
            else:
                new = proposed
            new = float(np.clip(new, self.min_val, self.max_val))
        else:
            new = float(np.clip(proposed, self.min_val, self.max_val))

        setattr(env.unwrapped, self.attribute, new)

        if verbose:
            print(f"[inc-reflect+noise] {self.attribute} step {step}: {cur:.6f}->{new:.6f} "
                  f"(Δ={change:+.6f}, sgn_now={sgn:+.0f}, sgn_persist={self._sgn:+.0f}, edge={self.edge_mode})")
        return new

def make_continuous_incrementer(*args, **kwargs):
    return _ContinuousIncrementer(*args, **kwargs)

# -----------------------------
# Sudden jump
# -----------------------------
class _SuddenJump:
    def __init__(self, attribute,
                 step_size_range=(0.2, 0.4),
                 interval_range=(10, 30),
                 min_val=0.0, max_val=1.0,
                 seed=None,
                 direction="both",
                 direction_prob=0.5,
                 edge_mode="clip",
                 debug=False,
                 **_ignored):  # swallow extras so the factory can pass uniform args
        assert direction in {"positive", "negative", "both"}
        assert edge_mode in {"clip", "reflect"}
        self.attribute = attribute
        self.step_size_range = (float(step_size_range[0]), float(step_size_range[1]))
        self.low, self.high = int(interval_range[0]), int(interval_range[1])
        if self.high < self.low:  # be strict here
            self.high = self.low
        self.min_val = float(min_val); self.max_val = float(max_val)
        self.seed = seed
        self.direction = direction
        self.direction_prob = float(direction_prob)
        self.edge_mode = edge_mode

        # state
        self.rng = None
        self._t = 0
        self.next_jump_step = None
        self.debug = bool(debug)

        self.config = {
            "type": "sudden_jump",
            "attribute": attribute,
            "step_size_range": step_size_range,
            "interval_range": (self.low, self.high),
            "min_val": min_val, "max_val": max_val, "seed": seed,
            "direction": direction, "direction_prob": direction_prob,
            "edge_mode": edge_mode,
        }

    def _sample_interval(self):
        # inclusive upper bound: [low, high]
        if self.low == self.high:
            return int(self.low)
        return int(self.rng.integers(self.low, self.high + 1))

    def _schedule_next(self):
        # schedule relative to *current* local time
        self.next_jump_step = self._t + self._sample_interval()
        if self.debug:
            print(f"[sudden_jump] t={self._t} -> next_jump_step={self.next_jump_step}")

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.rng = np.random.default_rng(_episode_seed(self.seed, episode_id, worker_id=worker_id))
        self._t = 0
        self._schedule_next()  # first jump  in [low, high] steps from reset

    def __call__(self, env, step, ctx, verbose=False):
        if self.rng is None or self.next_jump_step is None:
            self.reset(env, 0, ctx)

        cur = float(getattr(env.unwrapped, self.attribute))
        new = cur
        jumped = False

        while self._t >= self.next_jump_step:
            mag = float(self.rng.uniform(*self.step_size_range))
            if   self.direction == "positive": sgn = +1.0
            elif self.direction == "negative": sgn = -1.0
            else: sgn = +1.0 if self.rng.random() < self.direction_prob else -1.0

            proposed = new + sgn * mag
            if self.edge_mode == "clip":
                after = float(np.clip(proposed, self.min_val, self.max_val))
            else:
                if proposed < self.min_val:
                    over = self.min_val - proposed; after = self.min_val + over
                elif proposed > self.max_val:
                    over = proposed - self.max_val; after = self.max_val - over
                else:
                    after = proposed
                after = float(np.clip(after, self.min_val, self.max_val))

            # DEBUG: print full details
            if verbose:
                print(f"[sudden_jump] t={self._t} cur={cur:.6f} sgn={sgn:+.0f} "
                    f"mag={mag:.6f} proposed={proposed:.6f} -> after={after:.6f} "
                    f"(bounds[{self.min_val:.3f},{self.max_val:.3f}])")

            new = after
            jumped = True
            self.next_jump_step += int(self.rng.integers(self.low, self.high))

        if jumped:
            setattr(env.unwrapped, self.attribute, new)
            if verbose:
                print(f"[sudden_jump] t={self._t}: {cur:.4f} -> {new:.4f} (next @{self.next_jump_step})")

        self._t += 1
        return float(getattr(env.unwrapped, self.attribute))

def make_sudden_jump(*args, **kwargs):
    return _SuddenJump(*args, **kwargs)

# -----------------------------
# Random walk
# -----------------------------
class _RandomWalk:
    def __init__(self, attribute, std=0.01, min_val=0.0, max_val=1.0, seed=None,
                 **_ignored):  # swallow extras so the factory can pass uniform args
        self.attribute = attribute
        self.std = float(std)
        self.min_val = float(min_val); self.max_val = float(max_val)
        self.seed = seed; self.rng = None
        self.config = {"type":"random_walk","attribute":attribute,"std":std,
                       "min_val":min_val,"max_val":max_val,"seed":seed}

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.rng = np.random.default_rng(_episode_seed(self.seed, episode_id, worker_id=worker_id))

    def __call__(self, env, step, ctx, verbose=False):
        if self.rng is None: self.reset(env, 0, ctx)
        cur = float(getattr(env.unwrapped, self.attribute))
        new = float(np.clip(cur + float(self.rng.normal(0.0, self.std)), self.min_val, self.max_val))
        setattr(env.unwrapped, self.attribute, new)
        if verbose:
            print(f"[rw] {self.attribute} step {step}: {cur:.4f}->{new:.4f}")
        return new

def make_random_walk(*args, **kwargs):
    return _RandomWalk(*args, **kwargs)

# -----------------------------
# Piecewise constant (with optional bounds)
# -----------------------------
class _PiecewiseConstant:
    def __init__(
        self,
        attribute,
        values,
        interval_range=(30, 50),
        seed=None,
        min_val=-np.inf,
        max_val=np.inf,
        **_ignored,          # swallow extras so the factory can pass uniform args
    ):
        self.attribute = attribute
        self.values = list(values)
        self.low, self.high = int(interval_range[0]), int(interval_range[1])
        if self.high <= self.low:
            self.high = self.low + 1
        self.seed = seed
        self.min_val = float(min_val)
        self.max_val = float(max_val)

        self.rng = None
        self.current_value = None
        self.next_switch_step = None
        self.config = {
            "type": "piecewise_constant",
            "attribute": attribute,
            "values": list(values),
            "interval_range": (self.low, self.high),
            "seed": seed,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    def _pick_value(self):
        v = float(self.rng.choice(self.values))
        # clamp to bounds if provided
        return float(np.clip(v, self.min_val, self.max_val))

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.rng = np.random.default_rng(_episode_seed(self.seed, episode_id, worker_id=worker_id))
        self.current_value = self._pick_value()
        self.next_switch_step = int(self.rng.integers(self.low, self.high))
        setattr(env.unwrapped, self.attribute, self.current_value)

    def __call__(self, env, step, ctx, verbose=False):
        if self.rng is None or self.current_value is None:
            self.reset(env, 0, ctx)
        if step >= self.next_switch_step:
            self.current_value = self._pick_value()
            self.next_switch_step += int(self.rng.integers(self.low, self.high))
            if verbose:
                print(f"[pwc] {self.attribute} switch @ {step} -> {self.current_value}")
        setattr(env.unwrapped, self.attribute, self.current_value)
        return self.current_value

def make_piecewise_constant(*args, **kwargs):
    return _PiecewiseConstant(*args, **kwargs)

# -----------------------------
# Sinusoidal (reflecting bounds)
# -----------------------------
class _Sinusoidal:
    def __init__(self, attribute, amplitude=0.1, period=100, min_val=0.0, max_val=1.0,
                 seed=None, dir_sign=1, offset_from_context=True, **_ignored):
        self.attribute = attribute
        self.A = float(amplitude)
        self.w = 2*np.pi / max(1, period)
        self.min_val = float(min_val); self.max_val = float(max_val)
        self.seed = seed
        self.offset_from_context = bool(offset_from_context) #
        self.phase = 0.0

        self.dir_sign = dir_sign  # +1 or -1
        self.init_sign = dir_sign

        self.baseline = None
        self.config = {
            "type":"sinusoidal_additive_reflecting","attribute":attribute,
            "amplitude":amplitude,"period":period,"min_val":min_val,"max_val":max_val,
            "seed":seed,"offset_from_context":bool(offset_from_context),"reflecting_bounds":True,
        }

    def _capture_baseline(self, env, ctx):
        if self.offset_from_context and isinstance(ctx, dict) and self.attribute in ctx:
            return float(ctx[self.attribute])
        return float(getattr(env.unwrapped, self.attribute))

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.baseline = self._capture_baseline(env, ctx)
        if self.seed is None:
            self.phase = 0.0
            self.dir_sign = self.init_sign
        else:
            s = _episode_seed(self.seed, episode_id, worker_id=worker_id)
            rng = np.random.default_rng(s)
            self.dir_sign = 1 if rng.random() >= 0.5 else -1
            self.phase = 0.0
        base = float(np.clip(self.baseline, self.min_val, self.max_val))
        setattr(env.unwrapped, self.attribute, base)

    def __call__(self, env, step, ctx, verbose=False):
        if self.baseline is None: self.reset(env, 0, ctx)
        next_phase = self.phase + self.dir_sign * self.w
        nxt = self.baseline - self.A*np.sin(next_phase)
        # reflect at bounds by default
        if nxt < self.min_val or nxt > self.max_val:
            self.dir_sign *= -1
            next_phase = self.phase + self.dir_sign * self.w
        self.phase = next_phase
        val = float(np.clip(self.baseline - self.A*np.sin(self.phase), self.min_val, self.max_val))
        setattr(env.unwrapped, self.attribute, val)
        if verbose:
            ep = getattr(env, 'episode_count', '?')
            print(f"[sin] Ep {ep} step {step} {self.attribute}: base {self.baseline:.4f} -> {val:.4f}")
        return val

def make_sinusoidal(*args, **kwargs):
    return _Sinusoidal(*args, **kwargs)

# -----------------------------
# Cosine annealing (one-shot or with warm restarts)
# -----------------------------
class _CosineAnnealing:
    def __init__(self, attribute,
                start=None, end=None,
                T_max=1000,
                mode="once",
                T_0=200, T_mult=1,
                min_val=-np.inf, max_val=np.inf,
                offset_from_context=True,
                boundary_eps=0.05,
                neighborhood_radius=None,
                min_delta=0.0,
                direction="auto",
                seed=None,
                ping_pong=True,
                retarget="swap",
                repel_edge_eps_frac=0.05  # enforce-away threshold as % of band width
                ):
        assert mode in {"once", "cycle"}
        assert direction in {"auto", "random", "to_min", "to_max"}
        self.repel_edge_eps_frac = float(max(0.0, repel_edge_eps_frac))
        self.retarget = retarget # "swap" | "new_target" | "none"
        self.attribute = attribute
        self.start = None if start is None else float(start)
        self.end   = None if end   is None else float(end)
        self.T_max = int(max(1, T_max))
        self.mode = mode
        self.T_0 = int(max(1, T_0))
        self.T_mult = int(max(1, T_mult))
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.offset_from_context = bool(offset_from_context)
        self.boundary_eps = float(max(0.0, boundary_eps))
        self.neighborhood_radius = None if neighborhood_radius is None else float(max(0.0, neighborhood_radius))
        self.min_delta = float(max(0.0, min_delta))
        self.direction = direction
        self.seed = seed

        # runtime
        self._episode_started = False
        self._rng = None                   # np.random.Generator
        self._initial_start = None
        self._start_value = None
        self._end_value = None
        self._cycle_start_step = 0
        self._curr_cycle_len = self.T_0
        self._cycle_idx = 0
        self._band_low = self.min_val
        self._band_high = self.max_val

        self.config = {
            "type": "cosine_annealing",
            "attribute": attribute,
            "start": start, "end": end,
            "T_max": T_max, "mode": mode,
            "T_0": T_0, "T_mult": T_mult,
            "min_val": min_val, "max_val": max_val,
            "offset_from_context": bool(offset_from_context),
            "boundary_eps": self.boundary_eps,
            "neighborhood_radius": self.neighborhood_radius,
            "min_delta": self.min_delta,
            "direction": self.direction,
            "seed": seed,
            "retarget": self.retarget,
            "repel_edge_eps_frac": self.repel_edge_eps_frac,
        }

    # --- helpers -------------------------------------------------------------

    def _repel_edge_target(self, start_val):
        """If start is too close to a band edge, force target to the opposite edge.
        Returns: forced_target or None if no forcing is needed.
        """
        span = max(self._band_high - self._band_low, 1e-12)
        thr = self.repel_edge_eps_frac * span
        d_low  = abs(start_val - self._band_low)
        d_high = abs(self._band_high - start_val)
        if d_low <= thr:
            return self._band_high
        if d_high <= thr:
            return self._band_low
        return None

    def _capture_baseline(self, env, ctx):
        if self.offset_from_context and isinstance(ctx, dict) and self.attribute in ctx:
            return float(ctx[self.attribute])
        return float(getattr(env.unwrapped, self.attribute))

    def _set_band(self, center):
        if self.neighborhood_radius is None or not np.isfinite(self.neighborhood_radius) or self.neighborhood_radius <= 0.0:
            low, high = self.min_val, self.max_val
        else:
            low  = max(self.min_val, center - self.neighborhood_radius)
            high = min(self.max_val, center + self.neighborhood_radius)
            if low > high:
                low, high = self.min_val, self.max_val
        self._band_low, self._band_high = float(low), float(high)

    def _nearest_edge(self, x):
        d_min = abs(x - self._band_low)
        d_max = abs(self._band_high - x)
        span = max(self._band_high - self._band_low, 1e-12)
        thr = self.boundary_eps * span
        if d_min < d_max:
            return self._band_high if d_min <= thr else self._band_low
        else:
            return self._band_low if d_max <= thr else self._band_high

    def _pick_target_edge(self, start_val):
        if self.direction == "to_min": return self._band_low
        if self.direction == "to_max": return self._band_high
        if self.direction == "random": return self._band_low if self._rng.random() < 0.5 else self._band_high
        return self._nearest_edge(start_val)  # "auto"

    @staticmethod
    def _cosine_interp(start, end, t, T):
        if T <= 1: return float(start)
        t = float(np.clip(t, 0, T - 1))
        frac = t / (T - 1)
        return float(end + (start - end) * 0.5 * (1.0 + np.cos(np.pi * frac)))

    # --- lifecycle -----------------------------------------------------------

    def _init_cycle(self, center_start, force_new_target=False):
        # band centered at ORIGINAL start (hard-restart locality)
        self._set_band(self._initial_start)
        start_val = float(np.clip(center_start, self._band_low, self._band_high))

        # 1) if explicit end in once-mode (first cycle), keep that
        if self.mode == "once" and self.end is not None and self._cycle_idx == 0 and not force_new_target:
            end_val = float(np.clip(self.end, self._band_low, self._band_high))
        else:
            # 2) force direction away from edge if within repel threshold
            forced = self._repel_edge_target(start_val)
            if forced is not None:
                end_val = forced
            else:
                # 3) otherwise, pick as usual (random/auto/to_min/to_max) within the band
                end_val = float(np.clip(self._pick_target_edge(start_val), self._band_low, self._band_high))

        # optional safety: ensure minimum movement inside the band
        if self.min_delta > 0.0 and abs(end_val - start_val) < self.min_delta:
            sgn = +1.0 if (end_val >= start_val) else -1.0
            end_val = float(np.clip(start_val + sgn*self.min_delta, self._band_low, self._band_high))

        self._start_value, self._end_value = start_val, end_val


    def reset(self, env, episode_id, ctx=None, worker_id=0):
        s = _episode_seed(self.seed, episode_id, worker_id=worker_id)  # your seeding style
        self._rng = np.random.default_rng(s)

        base_start = self._capture_baseline(env, ctx) if self.start is None else float(self.start)
        base_start = float(np.clip(base_start, self.min_val, self.max_val))
        self._initial_start = base_start

        self._cycle_idx = 0
        self._curr_cycle_len = (self.T_max if self.mode == "once" else self.T_0)
        self._cycle_start_step = 0
        self._init_cycle(base_start)

        self._episode_started = True
        setattr(env.unwrapped, self.attribute, float(self._start_value))

    # --- stepping ------------------------------------------------------------

    def __call__(self, env, step, ctx, verbose=False):
        if not self._episode_started:
            self.reset(env, 0, ctx)

        if self.mode == "once":
            t = min(step, self.T_max)
            val = self._cosine_interp(self._start_value, self._end_value, t, self.T_max)
        else:
            while step - self._cycle_start_step >= self._curr_cycle_len:
                self._cycle_idx += 1
                self._cycle_start_step += self._curr_cycle_len
                self._curr_cycle_len = int(self._curr_cycle_len * self.T_mult)

                if self.retarget == "restart":
                    # Always hard reset to the original start (band recenters there)
                    self._init_cycle(self._initial_start, force_new_target=True)

                elif self.retarget == "random":
                    # Flip a coin: restart vs continue-from-end
                    if self._rng.random() < 0.5:
                        # hard reset
                        self._init_cycle(self._initial_start, force_new_target=True)
                    else:
                        # continue: start next cycle from the previous end and recenter the band there
                        next_start = float(np.clip(self._end_value, self.min_val, self.max_val))
                        self._set_band(next_start)  # recenter locality on new start
                        self._init_cycle(next_start, force_new_target=True)

                else:
                    # (optional) fallback — keep current behavior if someone passes an unexpected value
                    self._init_cycle(self._initial_start, force_new_target=True)

            t_in_cycle = step - self._cycle_start_step
            T = max(1, self._curr_cycle_len)
            val = self._cosine_interp(self._start_value, self._end_value, t_in_cycle, T)

            # Bounce within the local band
            if val > self._band_high:
                over = val - self._band_high
                val = self._band_high - over
            elif val < self._band_low:
                over = self._band_low - val
                val = self._band_low + over

        val = float(np.clip(val, self.min_val, self.max_val))
        setattr(env.unwrapped, self.attribute, val)

        if verbose:
            mi = f"[t={min(step, self.T_max)}/{self.T_max}]" if self.mode == "once" \
                 else f"[cycle len={self._curr_cycle_len} t={step - self._cycle_start_step}]"
            print(f"[cos-local HR] {mi} {self.attribute}: "
                  f"{self._start_value:.6g} -> {val:.6g} -> {self._end_value:.6g} "
                  f"(band[{self._band_low:.4f},{self._band_high:.4f}])")
        return val

def make_cosine_annealing(*args, **kwargs):
    return _CosineAnnealing(*args, **kwargs)


# ===== Stacking helper =====
class _SequentialComposite:
    """
    Compose multiple updaters for the same attribute.
    Calls them in order on each step. The last one's value is returned.
    """
    def __init__(self, attribute, parts):
        self.attribute = attribute
        self.parts = list(parts)
        self.config = {
            "type": "sequential_composite",
            "attribute": attribute,
            "parts": [getattr(p, "config", type(p).__name__) for p in parts],
        }

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        for p in self.parts:
            if hasattr(p, "reset"):
                p.reset(env, episode_id, ctx=ctx, worker_id=worker_id)

    def __call__(self, env, step, ctx, verbose=False):
        last = None
        for p in self.parts:
            last = p(env, step, ctx, verbose=verbose)
        return last

# -----------------------------
# CartPole coupling: mass from length (with reset forwarding)
# -----------------------------
def with_masspole_from_length(update_obj, min_masspole=0.01, max_masspole=np.inf):
    def _recompute(env):
        L = float(env.unwrapped.length)
        m = float(np.clip(L * _get_init_density(env), min_masspole, max_masspole))
        env.unwrapped.masspole = m
        env.unwrapped.polemass_length = m * L
        env.unwrapped.total_mass = m + env.unwrapped.masscart

    class _Wrapper:
        def __init__(self, inner):
            self.inner = inner
            self.config = {
                "type": "with_masspole_from_length",
                "min_masspole": min_masspole,
                "max_masspole": max_masspole,
                "inner_function": getattr(inner, "config", type(inner).__name__),
            }
        def reset(self, env, episode_id, ctx=None, worker_id=0):
            self.inner.reset(env, episode_id, ctx=ctx, worker_id=worker_id)
            _recompute(env)  # sync derived params at episode start
        def __call__(self, env, step, ctx, verbose=False):
            new_L = self.inner(env, step, ctx, verbose=verbose)
            _recompute(env)  # keep derived params in sync after each change
            return new_L

    return _Wrapper(update_obj)

def _get_init_density(env):
    # Preferred: search wrapper stack without warnings
    try:
        return float(env.get_wrapper_attr("init_density"))
    except Exception:
        pass

    # Next: read from base env (unwrapped)
    raw = getattr(env, "unwrapped", env)
    if hasattr(raw, "init_density"):
        return float(raw.init_density)

    # Legacy fallback (if someone set the private name)
    if hasattr(raw, "_init_density"):
        return float(raw._init_density)

    # Last resort: derive from current params
    if hasattr(raw, "masspole") and hasattr(raw, "length") and getattr(raw, "length") not in (None, 0):
        return float(raw.masspole / raw.length)

    raise AttributeError("init_density not set and cannot be derived")
