"""RECAP: reciprocal sparse attention and hierarchical caching for video diffusion.

Two levels of caching, scheduled online from measured errors:

* attention level: a dense refresh caches, per layer, a refinement of the attention tail
  over its coarse estimate; corrected sparse steps compute the selected blocks exactly
  and add the drift-weighted refinement to a fresh coarse estimate;
* prediction level: computed predictions renew a prediction cache, from which other
  steps are extrapolated without a transformer call.

Usage::

    from recap import load_config, install

    config = load_config("configs/cogvideox1.5_fast.json")
    session = install(pipe, config)
    for prompt in prompts:
        session.reset()
        video = pipe(prompt, num_inference_steps=config.schedule.num_steps)
"""

from __future__ import annotations

from .config import OperatingPoint, RecapConfig, config_from_dict, load_config
from .controller import Controller
from .operator import EXACT, REFRESH, SPARSE, Operator
from .session import REUSE, RecapSession

__all__ = [
    "Controller",
    "Operator",
    "OperatingPoint",
    "RecapConfig",
    "RecapSession",
    "config_from_dict",
    "install",
    "load_config",
    "EXACT",
    "REFRESH",
    "SPARSE",
    "REUSE",
]

_BUILDERS = {
    "cogvideox1.5": "recap.models.cogvideox",
    "wan2.1-1.3b": "recap.models.wan",
    "hunyuan": "recap.models.hunyuan",
}


def install(pipe, config: RecapConfig, *, model: str | None = None) -> RecapSession:
    """Install RECAP into ``pipe`` and return the session.  Call
    :meth:`RecapSession.reset` before every sample.
    """
    model = str(model or config.model)
    if model not in _BUILDERS:
        raise ValueError(f"no integration for model {model!r}; have {sorted(_BUILDERS)}")
    import importlib

    module = importlib.import_module(_BUILDERS[model])
    operator, info = module.build_operator(pipe, config.validate())
    session = RecapSession(
        config,
        operator,
        calls_per_step=int(info.get("calls_per_step", 1)),
        model_info=info,
    )
    session.install(pipe.transformer)
    return session
