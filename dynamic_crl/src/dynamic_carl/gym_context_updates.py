import numpy as np

# =============================================================================
# 1. CORE UTILITIES & ACTUATORS
# =============================================================================

def _episode_seed(base_seed, episode_id, worker_id=0):
    """Generates a deterministic 32-bit integer seed per episode."""
    bs = 0 if base_seed is None else int(base_seed)
    ss = np.random.SeedSequence([bs, int(episode_id), int(worker_id)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])

class Actuator:
    """Protocol for getting/setting values in an environment."""
    @property
    def name(self): raise NotImplementedError
    def get(self, env): raise NotImplementedError
    def set(self, env, val): raise NotImplementedError

class SimpleAttributeActuator(Actuator):
    """
    Default behavior: getattr/setattr on env.unwrapped.
    Used when 'attribute' is a string.
    """
    def __init__(self, attribute_name):
        self.attr = attribute_name

    @property
    def name(self): return self.attr

    def get(self, env):
        return float(getattr(env.unwrapped, self.attr))

    def set(self, env, val):
        val = float(val)
        setattr(env.unwrapped, self.attr, val)
        return val

# =============================================================================
# 2. BASE SCHEDULER
# =============================================================================

class _SchedBase:
    """
    Base class handling:
    1. Actuator instantiation (auto-detects string vs object)
    2. Bounds clipping (min_val/max_val)
    3. Configuration dictionary storage
    """
    def __init__(self, attribute, min_val=-np.inf, max_val=np.inf):
        # 1. Setup Actuator
        if isinstance(attribute, str):
            self.act = SimpleAttributeActuator(attribute)
        elif hasattr(attribute, "get") and hasattr(attribute, "set"):
            self.act = attribute
        else:
            raise ValueError(f"Attribute must be a string or an Actuator object, got {type(attribute)}")

        # 2. Setup Bounds
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        
        # 3. Base Config
        self.config = {
            "attribute": self.act.name,
            "min_val": min_val,
            "max_val": max_val
        }

    def _get(self, env):
        return self.act.get(env)

    def _set(self, env, val):
        # Universal clipping
        val = float(np.clip(val, self.min_val, self.max_val))
        return self.act.set(env, val)
    
    def reset(self, env, episode_id, ctx=None, worker_id=0):
        """Must be implemented by subclasses to reset internal RNG/state."""
        raise NotImplementedError

    def __call__(self, env, step, ctx, verbose=False):
        """Must be implemented by subclasses."""
        raise NotImplementedError


# =============================================================================
# 3. SCHEDULERS
# =============================================================================

# --- Identity ---
class _Identity(_SchedBase):
    def __init__(self, attribute, min_val=-np.inf, max_val=np.inf, seed=None):
        super().__init__(attribute, min_val, max_val)
        self.config.update({"type": "identity", "seed": seed})

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        # Ensure value is within bounds at start, if strictly required
        val = self._get(env)
        self._set(env, val)

    def __call__(self, env, step, ctx, verbose=False):
        val = self._get(env)
        if verbose:
            print(f"[id] {self.act.name}: {val:.6g}")
        return val

def make_identity(*args, **kwargs): return _Identity(*args, **kwargs)


# --- Continuous Incrementer ---
class _ContinuousIncrementer(_SchedBase):
    def __init__(self, attribute, delta=0.01, min_val=0.0, max_val=1.0,
                 direction="positive", direction_prob=0.5,
                 *, edge_mode="clip", episode_direction=None,
                 noise_std=0.0, noise_std_frac_of_delta=None,
                 follow_predefined_prob=None, seed=None):
        super().__init__(attribute, min_val, max_val)
        assert direction in {"positive", "negative", "both"}
        assert edge_mode in {"clip", "reflect"}
        if episode_direction: assert episode_direction in {"up", "down", "random"}

        self.delta = float(delta)
        self.direction = direction
        self.direction_prob = float(direction_prob)
        self.edge_mode = edge_mode
        self.episode_direction = episode_direction
        self.noise_std = float(noise_std)
        self.noise_std_frac = noise_std_frac_of_delta
        self.follow_prob = follow_predefined_prob
        self.seed = seed
        
        # State
        self.rng = None
        self._sgn = 1.0
        self._noise_eff = 0.0

        self.config.update({
            "type": "continuous_incrementer",
            "delta": delta, "direction": direction, "edge_mode": edge_mode,
            "seed": seed, "episode_direction": episode_direction
        })

    def _choose_init_sign(self):
        if self.episode_direction == "up": return 1.0
        if self.episode_direction == "down": return -1.0
        if self.episode_direction == "random": return 1.0 if self.rng.random() >= 0.5 else -1.0
        if self.direction == "positive": return 1.0
        if self.direction == "negative": return -1.0
        return 1.0 if self.rng.random() < self.direction_prob else -1.0

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.rng = np.random.default_rng(_episode_seed(self.seed, episode_id, worker_id))
        self._sgn = self._choose_init_sign()
        self._noise_eff = abs(self.noise_std_frac * self.delta) if self.noise_std_frac else abs(self.noise_std)

    def __call__(self, env, step, ctx, verbose=False):
        if self.rng is None: self.reset(env, 0, ctx)
        cur = self._get(env)

        # Determine step direction
        sgn_persist = self._sgn
        # If stochastic direction allowed:
        if self.edge_mode != "reflect" and self.episode_direction is None and self.direction == "both":
            sgn_persist = 1.0 if self.rng.random() < self.direction_prob else -1.0
        
        if self.follow_prob is not None:
             sgn = sgn_persist if self.rng.random() < self.follow_prob else -sgn_persist
        else:
             sgn = sgn_persist

        change = sgn * self.delta
        if self._noise_eff > 0:
            change += float(self.rng.normal(0, self._noise_eff))

        proposed = cur + change
        
        # Handle Edges
        if self.edge_mode == "reflect":
            if proposed < self.min_val:
                proposed = self.min_val + (self.min_val - proposed)
                self._sgn *= -1.0
            elif proposed > self.max_val:
                proposed = self.max_val - (proposed - self.max_val)
                self._sgn *= -1.0
        
        new_val = self._set(env, proposed) # _set handles clipping
        
        if verbose:
            print(f"[inc] {self.act.name} {step}: {cur:.4f}->{new_val:.4f} (d={change:+.4f})")
        return new_val

def make_continuous_incrementer(*args, **kwargs): return _ContinuousIncrementer(*args, **kwargs)


# --- Sudden Jump ---
class _SuddenJump(_SchedBase):
    def __init__(self, attribute, step_size_range=(0.2, 0.4), interval_range=(10, 30),
                 min_val=0.0, max_val=1.0, seed=None, direction="both",
                 direction_prob=0.5, edge_mode="clip", debug=False):
        super().__init__(attribute, min_val, max_val)
        self.mag_range = step_size_range
        self.low, self.high = int(interval_range[0]), max(int(interval_range[0]), int(interval_range[1]))
        self.seed = seed
        self.direction = direction
        self.direction_prob = float(direction_prob)
        self.edge_mode = edge_mode
        
        self.rng = None
        self._t = 0
        self.next_jump = None
        
        self.config.update({"type": "sudden_jump", "interval": interval_range, "mag": step_size_range})

    def _get_interval(self):
        return int(self.low) if self.low == self.high else int(self.rng.integers(self.low, self.high + 1))

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.rng = np.random.default_rng(_episode_seed(self.seed, episode_id, worker_id))
        self._t = 0
        self.next_jump = self._get_interval()

    def __call__(self, env, step, ctx, verbose=False):
        if self.rng is None: self.reset(env, 0, ctx)
        cur = self._get(env)
        new = cur
        jumped = False
        
        while self._t >= self.next_jump:
            mag = float(self.rng.uniform(*self.mag_range))
            if self.direction == "positive": sgn = 1.0
            elif self.direction == "negative": sgn = -1.0
            else: sgn = 1.0 if self.rng.random() < self.direction_prob else -1.0
            
            proposed = new + sgn * mag
            
            # Helper for reflection logic since _set only clips
            if self.edge_mode == "reflect":
                if proposed < self.min_val: proposed = self.min_val + (self.min_val - proposed)
                elif proposed > self.max_val: proposed = self.max_val - (proposed - self.max_val)
            
            new = proposed
            jumped = True
            self.next_jump += self._get_interval()
        
        if jumped:
            new = self._set(env, new)
            if verbose: print(f"[jump] {self.act.name} t={self._t}: {cur:.4f}->{new:.4f}")
            
        self._t += 1
        return self._get(env) # Return current state

def make_sudden_jump(*args, **kwargs): return _SuddenJump(*args, **kwargs)


# --- Random Walk ---
class _RandomWalk(_SchedBase):
    def __init__(self, attribute, std=0.01, min_val=0.0, max_val=1.0, seed=None):
        super().__init__(attribute, min_val, max_val)
        self.std = std
        self.seed = seed
        self.rng = None
        self.config.update({"type": "random_walk", "std": std, "seed": seed})

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.rng = np.random.default_rng(_episode_seed(self.seed, episode_id, worker_id))

    def __call__(self, env, step, ctx, verbose=False):
        if self.rng is None: self.reset(env, 0, ctx)
        cur = self._get(env)
        noise = float(self.rng.normal(0, self.std))
        new = self._set(env, cur + noise)
        if verbose: print(f"[rw] {self.act.name} {step}: {cur:.4f}->{new:.4f}")
        return new

def make_random_walk(*args, **kwargs): return _RandomWalk(*args, **kwargs)


# --- Piecewise Constant ---
class _PiecewiseConstant(_SchedBase):
    def __init__(self, attribute, values, interval_range=(30, 50), seed=None,
                 min_val=-np.inf, max_val=np.inf):
        super().__init__(attribute, min_val, max_val)
        self.values = list(values)
        self.low, self.high = int(interval_range[0]), max(int(interval_range[0]), int(interval_range[1]) + 1)
        self.seed = seed
        self.rng = None
        self.curr_val = None
        self.next_switch = None
        self.config.update({"type": "piecewise", "values": values})

    def _pick(self):
        val = float(self.rng.choice(self.values))
        return val # Clipping handled by _set

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.rng = np.random.default_rng(_episode_seed(self.seed, episode_id, worker_id))
        self.curr_val = self._pick()
        self.next_switch = int(self.rng.integers(self.low, self.high))
        self._set(env, self.curr_val)

    def __call__(self, env, step, ctx, verbose=False):
        if self.rng is None: self.reset(env, 0, ctx)
        if step >= self.next_switch:
            self.curr_val = self._pick()
            self.next_switch += int(self.rng.integers(self.low, self.high))
            if verbose: print(f"[pwc] {self.act.name} switch@{step}->{self.curr_val}")
        
        return self._set(env, self.curr_val)

def make_piecewise_constant(*args, **kwargs): return _PiecewiseConstant(*args, **kwargs)


# --- Sinusoidal ---
class _Sinusoidal(_SchedBase):
    def __init__(self, attribute, amplitude=0.1, period=100, min_val=0.0, max_val=1.0,
                 seed=None, dir_sign=1, offset_from_context=True):
        super().__init__(attribute, min_val, max_val)
        self.A = float(amplitude)
        self.w = 2 * np.pi / max(1, period)
        self.seed = seed
        self.offset = bool(offset_from_context)
        self.init_sign = int(np.sign(dir_sign) or 1)
        
        self.dir_sign = 1
        self.phase = 0.0
        self.baseline = None
        self.config.update({"type": "sinusoidal", "A": amplitude, "period": period})

    def _get_baseline(self, env, ctx):
        # If the value is in context, prefer that as the "center" of the oscillation
        if self.offset and isinstance(ctx, dict) and self.act.name in ctx:
            return float(ctx[self.act.name])
        return self._get(env)

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.baseline = self._get_baseline(env, ctx)
        if self.seed is None:
            self.phase = 0.0
            self.dir_sign = self.init_sign
        else:
            rng = np.random.default_rng(_episode_seed(self.seed, episode_id, worker_id))
            self.dir_sign = 1 if rng.random() >= 0.5 else -1
            self.phase = 0.0
        
        # Start at baseline
        self._set(env, self.baseline)

    def __call__(self, env, step, ctx, verbose=False):
        if self.baseline is None: self.reset(env, 0, ctx)
        
        # Calculate next position to check bounds
        next_phase = self.phase + self.dir_sign * self.w
        val = self.baseline - self.A * np.sin(next_phase)
        
        # Reflect logic
        if val < self.min_val or val > self.max_val:
            self.dir_sign *= -1
            next_phase = self.phase + self.dir_sign * self.w
        
        self.phase = next_phase
        final_val = self.baseline - self.A * np.sin(self.phase)
        return self._set(env, final_val)

def make_sinusoidal(*args, **kwargs): return _Sinusoidal(*args, **kwargs)


# --- Cosine Annealing ---
class _CosineAnnealing(_SchedBase):
    def __init__(self, attribute, start=None, end=None, T_max=1000, mode="once",
                 T_0=200, T_mult=1, min_val=-np.inf, max_val=np.inf,
                 offset_from_context=True, boundary_eps=0.05, neighborhood_radius=None,
                 min_delta=0.0, direction="auto", seed=None, retarget="swap",
                 repel_edge_eps_frac=0.05):
        super().__init__(attribute, min_val, max_val)
        
        self.start = float(start) if start is not None else None
        self.end = float(end) if end is not None else None
        self.T_max = max(1, int(T_max))
        self.mode = mode
        self.T_0 = max(1, int(T_0))
        self.T_mult = max(1, int(T_mult))
        
        self.offset = bool(offset_from_context)
        self.boundary_eps = boundary_eps
        self.radius = neighborhood_radius
        self.min_delta = min_delta
        self.direction = direction
        self.seed = seed
        self.retarget = retarget
        self.repel_frac = repel_edge_eps_frac
        
        # Runtime
        self.rng = None
        self._initial_start = None
        self._start_val = None
        self._end_val = None
        self._cycle_start = 0
        self._cycle_len = self.T_0
        self._band = (self.min_val, self.max_val)
        
        self.config.update({"type": "cosine", "mode": mode, "T_max": T_max})

    def _get_baseline(self, env, ctx):
        if self.offset and isinstance(ctx, dict) and self.act.name in ctx:
            return float(ctx[self.act.name])
        return self._get(env)

    def _update_band(self, center):
        if not self.radius or self.radius <= 0:
            self._band = (self.min_val, self.max_val)
        else:
            low = max(self.min_val, center - self.radius)
            high = min(self.max_val, center + self.radius)
            self._band = (low, high)

    def _init_cycle(self, start_center, force_target=False):
        self._update_band(start_center)
        band_low, band_high = self._band
        
        start_val = np.clip(start_center, band_low, band_high)
        
        # Determine End Value
        if self.mode == "once" and self.end is not None and not force_target:
            end_val = np.clip(self.end, band_low, band_high)
        else:
            # Logic to repel from edges or pick random target
            span = max(band_high - band_low, 1e-12)
            d_low = abs(start_val - band_low)
            d_high = abs(band_high - start_val)
            thr = self.repel_frac * span
            
            forced = None
            if d_low <= thr: forced = band_high
            elif d_high <= thr: forced = band_low
            
            if forced is not None:
                end_val = forced
            else:
                # Direction logic
                if self.direction == "to_min": target = band_low
                elif self.direction == "to_max": target = band_high
                elif self.direction == "random": target = band_low if self.rng.random() < 0.5 else band_high
                else: # auto - nearest edge
                     target = band_high if d_low < d_high else band_low
                end_val = target

        # Min delta check
        if self.min_delta > 0 and abs(end_val - start_val) < self.min_delta:
            sgn = 1.0 if end_val >= start_val else -1.0
            end_val = np.clip(start_val + sgn * self.min_delta, band_low, band_high)

        self._start_val = float(start_val)
        self._end_val = float(end_val)

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.rng = np.random.default_rng(_episode_seed(self.seed, episode_id, worker_id))
        
        base = self._get_baseline(env, ctx) if self.start is None else self.start
        self._initial_start = float(np.clip(base, self.min_val, self.max_val))
        
        self._cycle_start = 0
        self._cycle_len = self.T_max if self.mode == "once" else self.T_0
        
        self._init_cycle(self._initial_start)
        self._set(env, self._start_val)

    def __call__(self, env, step, ctx, verbose=False):
        if self.rng is None: self.reset(env, 0, ctx)
        
        if self.mode == "cycle":
            while step - self._cycle_start >= self._cycle_len:
                self._cycle_start += self._cycle_len
                self._cycle_len = int(self._cycle_len * self.T_mult)
                
                # Retarget logic
                if self.retarget == "swap":
                    # Center band on where we ended up
                    self._init_cycle(self._end_val, force_target=True)
                elif self.retarget == "random":
                    # Flip coin: hard restart vs swap
                    if self.rng.random() < 0.5:
                        self._init_cycle(self._initial_start, force_target=True)
                    else:
                        self._init_cycle(self._end_val, force_target=True)
                else: # restart
                    self._init_cycle(self._initial_start, force_target=True)

        t = step - self._cycle_start
        T = self._cycle_len
        
        # Cosine Interp
        if T <= 1: 
            val = self._start_val
        else:
            frac = np.clip(t, 0, T-1) / (T - 1)
            val = self._end_val + (self._start_val - self._end_val) * 0.5 * (1 + np.cos(np.pi * frac))
        
        # Bounce if interp overshoot (floating point errors or logic)
        bl, bh = self._band
        if val > bh: val = bh - (val - bh)
        if val < bl: val = bl + (bl - val)
        
        out = self._set(env, val)
        
        if verbose:
            print(f"[cos] {self.act.name} {out:.4f} (cycle {self._start_val:.2f}->{self._end_val:.2f})")
        return out

def make_cosine_annealing(*args, **kwargs): return _CosineAnnealing(*args, **kwargs)


# =============================================================================
# 4. COMPOSITES & WRAPPERS
# =============================================================================

class _SequentialComposite:
    """Stacks multiple updaters for the same attribute."""
    def __init__(self, attribute, parts):
        self.attribute = attribute
        self.parts = list(parts)
        self.config = {"type": "sequential", "parts": [p.config for p in parts]}

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        for p in self.parts:
            if hasattr(p, "reset"):
                p.reset(env, episode_id, ctx, worker_id)

    def __call__(self, env, step, ctx, verbose=False):
        last = None
        for p in self.parts:
            last = p(env, step, ctx, verbose)
        return last

def make_sequential(attribute, parts):
    return _SequentialComposite(attribute, parts)


# --- Dependency Wrapper (Generic) ---
class DependentAttributeWrapper:
    """
    Generic wrapper for handling attributes that depend on the scheduled one.
    Example: When 'length' changes, update 'mass' automatically.
    """
    def __init__(self, scheduler, callback_fn):
        self.inner = scheduler
        self.callback = callback_fn
        self.config = self.inner.config.copy()
        self.config["wrapper"] = "dependent_attribute"

    def reset(self, env, episode_id, ctx=None, worker_id=0):
        self.inner.reset(env, episode_id, ctx, worker_id)
        self.callback(env) # Sync immediately after reset

    def __call__(self, env, step, ctx, verbose=False):
        val = self.inner(env, step, ctx, verbose)
        self.callback(env) # Sync after update
        return val

# --- Example of Specific Dependency (CartPole) ---
def _cartpole_mass_updater(env):
    """Callback to update CartPole mass based on length."""
    raw = env.unwrapped
    if not hasattr(raw, "length"): return
    
    # Try to find init_density
    rho = getattr(raw, "init_density", None)
    if rho is None:
        # Fallback: derive from current state if valid
        if raw.length > 0: rho = raw.masspole / raw.length
        else: rho = 0.1 # Dangerous fallback
        
    L = float(raw.length)
    m = L * rho
    raw.masspole = m
    raw.polemass_length = m * L
    raw.total_mass = m + raw.masscart

def with_masspole_from_length(update_obj):
    return DependentAttributeWrapper(update_obj, _cartpole_mass_updater)