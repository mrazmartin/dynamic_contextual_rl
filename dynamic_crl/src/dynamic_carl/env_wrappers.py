from gymnasium import Env
from gymnasium.spaces import Dict as SpaceDict
from numpy import info
from gymnasium import spaces
from carl.context.context_space import NumericalContextFeature

class GymDynamicContextCarlWrapper(Env):
    def __init__(
        self,
        env,
        feature_update_fns,
        ctx_getters=None,     # maps key -> fn(env)->value
        ctx_setters=None,     # maps key -> fn(env, value)
        # ctx_to_change=None,   # which context can change, per episode or dynamically
        ctx_to_observe: set[str]=None,  # subset of context keys to expose to the agent
        worker_id=0,
        observe_context_mode: str = "live",  # "live" | "initial" | "none"
        mutate_obs_space: bool = False,      # drop "context" key from space when mode == "none"
        dctx_features_definitions=None, # carl like ContextFeature definitions for new dynamic ctxs
        verbose=False,
    ):
        """
        Wrapper for a Gym environment that adds dynamic context features.
        Args:
            env: The base Gym environment to wrap.
            feature_update_fns: Dict mapping context keys to functions that update them.
                Each update_fn has 4 arguments: (env, step_id, current_ctx_dict, verbose)
            ctx_getters: Optional dict mapping context keys to functions fn(env) -> value.
                Used to read current context values from the env.
            ctx_setters: Optional dict mapping context keys to functions fn(env, value).
                Used to set context values in the env.
            # ctx_to_change: Optional list of context keys to change over time.
            ctx_to_observe: Optional subset of context keys to expose to the agent (both dynamic and static).
                If None, expose all context keys (static + dynamic).
            observe_context_mode: "live" | "initial" | "none".
            mutate_obs_space: If True and mode == "none", drop the "context" space.
        """
        self.env = env
        self.worker_id = worker_id

        # dynamic context handling
        self.feature_update_fns = feature_update_fns or {}
        self.dctx_features_definitions = dctx_features_definitions or {}
        # dynamic ctx manipulation
        self.ctx_getters = ctx_getters or {}
        self.ctx_setters = ctx_setters or {}

        # self.ctx_to_change = list(ctx_to_change) if ctx_to_change else []

        self.ctx_to_observe = set(ctx_to_observe) if ctx_to_observe is not None else None
        self.verbose = verbose

        self._pending_seed = None # this is to be set upon first reset of the env to enforce seeding

        if observe_context_mode not in {"live", "initial", "none"}:
            raise ValueError(
                f"observe_context_mode must be one of {{'live','initial','none'}}, got {observe_context_mode}"
            )
        self.observe_context_mode = observe_context_mode
        
        # check whether each dynamic ctx has a getter and setter
        self._check_args() # validate getters/setters for dynamic contexts


        # Validate feature keys (OUTDATED - some dynamic contexts may not be in env.contexts)
        # context_keys = (
        #    env.contexts[0].keys()
        #    if hasattr(env, "contexts")
        #    else set(self.ctx_getters.keys()) | set(self.ctx_setters.keys())
        # )
        # for feature in self.feature_update_fns:
        #    if feature not in context_keys:
        #        raise ValueError(f"Feature '{feature}' not found in context keys: {context_keys}")

        # take over the action space
        self.action_space = env.action_space
        
        # Optionally mutate observation_space if context will never be exposed
        # NOTE: this is important as empty 'context' dicts can cause issues can break flatten wrappers
        if mutate_obs_space and self.observe_context_mode == "none":
            # if we don't want to observe context at all, remove it from the space
            if isinstance(self.env.observation_space, SpaceDict) and \
                    "context" in self.env.observation_space.spaces:
                new_spaces = dict(self.env.observation_space.spaces)
                new_spaces.pop("context", None)
                self.observation_space = SpaceDict(new_spaces)
            else:
                self.observation_space = env.observation_space
                if self.verbose:
                    print("[Wrapper] Cannot remove 'context' from observation_space; leaving as is.")
        else:
            # NOTE: if we don't remove the context key from observations, make sure dynamic ctxs are included
            # also make sure that the space won't include non-observable or default contexts
            self._update_obs_space()

        self.episode_count = -1
        self.step_count = 0
        self._ctx_reset_snapshot = {}

    def _dctx_feature_to_gym_space(self, cf_name):
        """
        Convert a dynamic context feature name to a Gym space.
        """
        context_feature = self.dctx_features_definitions[cf_name]
        if isinstance(context_feature, NumericalContextFeature):
            return spaces.Box(
                low=context_feature.lower, high=context_feature.upper
            )
        else:
            return spaces.Discrete(
                len(context_feature.choices)
            )

    def _update_obs_space(self):
        """
        Update the observation space to include the dynamic contexts.
        Also set self.ctx_to_observe if it was None.
        """
        
        # old/default observation of the CarlGymEnv space -> env.observation_space
        
        # 0. get the default observation space from the env
        base_obs_space = self.env.observation_space['obs']
        default_ctx_space = self.env.observation_space['context']

        # 1. get the keys of the context that should be observable
        # i) if no keys are specified, we have to take the whole context space + dynamic_ctxs
        if self.ctx_to_observe is None:
            static_ctx_keys = self.env.observation_space['context'].keys()
            dynamic_ctx_keys = set(self.feature_update_fns.keys())

            # union of both sets
            self.ctx_to_observe = static_ctx_keys | dynamic_ctx_keys

        # ii) if keys are specified, we don't have to do anything
        else:
            pass

        # 2. build the new context space for each self.ctx_to_observe key
        new_ctx_obs_spaces = {}
        for key in self.ctx_to_observe:
            # if the key is already in the default context space, take it from there
            if key in default_ctx_space.spaces:
                new_ctx_obs_spaces[key] = default_ctx_space.spaces[key]
            # else, we have to do it manually (for dynamic contexts)
            else:
                new_ctx_obs_spaces[key] = self._dctx_feature_to_gym_space(key)

        self.observation_space = SpaceDict({
            'obs': base_obs_space,
            'context': SpaceDict(new_ctx_obs_spaces)
        })

    def _check_args(self):
        # make sure that getters and setters are provided for all dynamic contexts
        for key in self.feature_update_fns.keys():
            if self.observe_context_mode == 'live' and key not in self.ctx_getters:
                raise ValueError(f"No getter provided for dynamic context '{key}'")
            if key not in self.ctx_setters:
                raise ValueError(f"No setter provided for dynamic context '{key}'")

    def _snapshot_ctx(self):
        # just check if some of them are corrupted into None somehow
        # if not (self.ctx_to_change or self.ctx_to_observe):
        if not self.ctx_to_observe:
            keys = self.ctx_getters.keys()
        else:
            # take union of both sets
            keys = set(self.ctx_getters.keys())
            # if self.ctx_to_change:
            #    keys |= set(self.ctx_to_change)
            if self.ctx_to_observe:
                keys |= set(self.ctx_to_observe)

        out = {}
        for key in keys:
            getter = self.ctx_getters.get(key, None)
            if getter is None:
                out[key] = getattr(self.env.unwrapped, key, None)
            else:
                out[key] = getter(self.env)
        return out

    def _apply_ctx_updates(self, updates: dict):
        for k, v in updates.items():
            setter = self.ctx_setters.get(k, None)
            if setter is not None:
                setter(self.env, v)
            else:
                if hasattr(self.env.unwrapped, k):
                    setattr(self.env.unwrapped, k, v)
                elif self.verbose:
                    print(f"[Wrapper] No setter for '{k}', and no env.unwrapped.{k}")

    def _filter_ctx_keys(self, ctx_dict):
        """
        When correctly initialized, this should not filter anything out.
        -> fallback safety only
        """
        if self.ctx_to_observe is None:
            return dict(ctx_dict)
        return {k: v for k, v in ctx_dict.items() if k in self.ctx_to_observe}

    def _make_observed_context(self, true_ctx):
        """
        Return either
         - the real state of the context (live),
         - the initial state of the context (initial),
         - or None (no context observed).

        This is mainly support for the initial mode for unobserved dynamics of known contexts.
        """
        mode = self.observe_context_mode
        if mode == "live":
            return self._filter_ctx_keys(true_ctx)
        if mode == "initial":
            return self._filter_ctx_keys(self._ctx_reset_snapshot)
        if mode == "none":
            return None
        raise ValueError(f"Unknown observe_context_mode '{mode}'")

    def _patch_obs_context(self, obs, true_ctx):
        # Only handle dict observations that contain "context"
        if not (isinstance(obs, dict) and "context" in obs):
            return obs

        observed_ctx = self._make_observed_context(true_ctx)

        if observed_ctx is None:
            # Keep key, but make it empty so downstream flatteners can handle it
            obs = dict(obs)
            obs["context"] = {}
            return obs

        # IMPORTANT: the dynamic changes are not caught by CARL 'context' observation
        # -> we have to patch them in manually here
        if isinstance(obs["context"], dict):
            for k in list(obs["context"].keys()):
                if k in observed_ctx and observed_ctx[k] is not None:
                    obs["context"][k] = float(observed_ctx[k])
                else:
                    # If not observed, remove it to avoid implying information
                    # another part of the safety net to filter out unobserved contexts
                    # NOTE: if the carl envs was properly set, none of these should be present...
                    obs["context"].pop(k, None)
        else:
            if self.verbose:
                print("[Wrapper] obs['context'] not dict; leaving as is.")
        return obs

    def set_next_seed(self, seed: int | None):
        self._pending_seed = None if seed is None else int(seed)

    def reset(self, *, seed=None, options=None, **kwargs):
        # If caller (e.g., SB3) passes seed=None, use our pending one once.
        if seed is None and self._pending_seed is not None:
            seed = self._pending_seed
            self._pending_seed = None

        self.step_count = 0
        self.episode_count += 1

        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        ctx_now = self._snapshot_ctx()
        self._ctx_reset_snapshot = dict(ctx_now)

        for fn in self.feature_update_fns.values():
            if hasattr(fn, "reset"):
                fn.reset(self.env, self.episode_count, ctx=ctx_now, worker_id=self.worker_id)
            else:
                raise ValueError("Feature update function missing 'reset' method.")

        obs = self._patch_obs_context(obs, ctx_now)
        return obs, info

    def step(self, action):
        if self.step_count == 0:
            obs, reward, terminated, truncated, info = self.env.step(action)
            true_ctx = self._snapshot_ctx()
            obs = self._patch_obs_context(obs, true_ctx)
            self.step_count += 1
            return obs, reward, terminated, truncated, info

        to_update_ctx = {
            k: self.ctx_getters.get(k, lambda e: getattr(e.unwrapped, k, None))(self.env)
            for k in self.feature_update_fns
        }
        for key, update_fn in self.feature_update_fns.items():
            res = update_fn(self.env, self.step_count, to_update_ctx, self.verbose)
            if isinstance(res, dict):
                self._apply_ctx_updates(res)
            else:
                self._apply_ctx_updates({key: res})

        obs, reward, terminated, truncated, info = self.env.step(action)
        true_ctx = self._snapshot_ctx()
        obs = self._patch_obs_context(obs, true_ctx)

        self.step_count += 1
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def get_true_context(self):
        """Return a snapshot of the real env context right now."""
        return self._snapshot_ctx()

    def get_observed_context_from_obs(self, obs, info=None):
        """
        Return the context as seen by the agent for this obs.
        Never infer from env when it is not present in obs.
        """
        if isinstance(obs, dict):
            if "context" in obs:
                return obs["context"]
            for k in ("ctx", "context_features"):
                if k in obs:
                    return obs[k]
        if isinstance(info, dict) and "context" in info:
            # only if your env explicitly uses info to expose observed context
            return info["context"]
        return None
