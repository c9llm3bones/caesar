"""caesar."""

from importlib import import_module

from .config import (
    AutoencoderConfig,
    EncoderConfig,
    DecoderConfig,
    create_encoder_config,
    create_decoder_config,
    create_autoencoder_config,
)


__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "encoder": ("caesar.modules.encoder", None),
    "decoder": ("caesar.modules.decoder", None),
    "autoencoder": ("caesar.modules.autoencoder", None),
    "Encoder": ("caesar.modules.encoder", "Encoder"),
    "Decoder": ("caesar.modules.decoder", "Decoder"),
    "Autoencoder": ("caesar.modules.autoencoder", "StructureAutoencoder"),
    "Vec3Array": ("caesar.utils.geometry", "Vec3Array"),
    "distance_rbf": ("caesar.modules.utils.geometry", "distance_rbf"),
    "index_mean": ("caesar.modules.utils.geometry", "index_mean"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "AutoencoderConfig",
    "EncoderConfig", 
    "DecoderConfig",
    "create_encoder_config",
    "create_decoder_config",
    "create_autoencoder_config",
    "Encoder",
    "Decoder",
    "Autoencoder",
    "Vec3Array",
    "distance_rbf",
    "index_mean",
    "encoder",
    "decoder",
    "autoencoder",
]
