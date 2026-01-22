from pathlib import Path
from typing import Any, Dict, Sequence, Callable, List, Optional, Tuple
import dynamic_crl.src.dynamic_carl.gym_context_updates as upd

def build_update_fn_from_spec(
    spec: Dict[str, Any],
    min_value: float,
    max_value: float,
    *,
    attribute: str,
) -> Callable[..., Any]:
    dyn_kind = spec.get("dyn_kind", "sinusoidal")
    fixed = spec.get("fixed", {}) or {}
    dyn_seed = spec.get("dyn_seed", None)

    if dyn_kind == "sinusoidal":
        frac = float(fixed.get("amplitude_frac", 0.0))
        period = int(fixed.get("period", 0))
        offset_from_context = bool(fixed.get("offset_from_context", True))
        amp = frac * (max_value - min_value) / 2.0

        return upd.make_sinusoidal(
            attribute=attribute,
            amplitude=amp,
            period=period,
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
            dir_sign=+1,
            offset_from_context=offset_from_context,
        )

    elif dyn_kind == "continuous_incrementer":
        delta_frac = float(fixed["delta_frac"])
        delta = delta_frac * (max_value - min_value)
        direction = fixed.get("direction", "both")
        edge_mode = fixed.get("edge_mode", "reflect")
        episode_direction = fixed.get("episode_direction", "random")
        follow_predefined_prob = float(fixed.get("follow_predefined_prob", 0.8))

        return upd.make_continuous_incrementer(
            attribute=attribute,
            delta=delta,                      # <--- IMPORTANT: scheduler expects `delta`
            min_val=min_value,
            max_val=max_value,
            seed=dyn_seed,
            direction=direction,
            edge_mode=edge_mode,
            episode_direction=episode_direction,
            follow_predefined_prob=follow_predefined_prob,
        )

    elif dyn_kind == "cosine_annealing":
        T_0 = int(fixed["T_0"])
        T_mult = int(fixed.get("T_mult", 1))
        mode = fixed.get("mode", "cycle")

        # YAML uses a *fraction* of the full [min_value, max_value] span
        neighborhood_radius_frac = float(fixed.get("neighborhood_radius_frac", 0.0))
        span = (max_value - min_value)
        neighborhood_radius = (
            0.5 * neighborhood_radius_frac * span
            if neighborhood_radius_frac > 0.0
            else None
        )

        direction = fixed.get("direction", "auto")
        offset_from_context = bool(fixed.get("offset_from_context", True))
        retarget = fixed.get("retarget", "restart")

        return upd.make_cosine_annealing(
            attribute=attribute,
            T_0=T_0,
            T_mult=T_mult,
            mode=mode,
            min_val=min_value,
            max_val=max_value,
            offset_from_context=offset_from_context,
            neighborhood_radius=neighborhood_radius,   # <--- IMPORTANT: correct kwarg
            direction=direction,
            seed=dyn_seed,
            retarget=retarget,
        )

    else:
        raise ValueError(f"Unknown dyn_kind: {dyn_kind}")
