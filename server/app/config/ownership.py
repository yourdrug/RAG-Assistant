"""Ownership markers for settings fields.

Every non-dynamic field of a section is annotated with exactly one marker:

* ``Secret`` — credentials, typed ``SecretStr``; read only from the process
  environment or ``.env.secrets`` (``Settings`` rejects them in ``.env``);
* ``Static[T]`` — deployment configuration, only from env/``.env``; never hot-reloaded;
* ``Metadata[T]`` — process metadata computed at startup.

``Secret`` fixes the type: ``repr()``, ``str()`` and ``model_dump()`` show a mask, and
only an explicit ``.get_secret_value()`` returns the credential. ``model_dump(mode="json")``
is masked too; never pass ``get_secret_value()`` results to logs. ``Static`` and
``Metadata`` are metadata only.

Unmarked fields are owned by ``application.runtime.dynamic_params.DYNAMIC_PARAMS``.
``application.runtime.setting_keys`` mirrors these markers (Application must not import
``config``); ``tests/test_configuration_sources.py`` keeps both in sync.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Annotated, TypeVar

from pydantic import SecretStr


if TYPE_CHECKING:
    from pydantic import BaseModel


class SettingOwnership(StrEnum):
    SECRET = auto()
    STATIC = auto()
    METADATA = auto()


T = TypeVar("T")

# Not generic: a credential is always ``SecretStr``, never a plain ``str``.
Secret = Annotated[SecretStr, SettingOwnership.SECRET]
Static = Annotated[T, SettingOwnership.STATIC]
Metadata = Annotated[T, SettingOwnership.METADATA]


def field_ownership(model: type[BaseModel], name: str) -> SettingOwnership | None:
    """The marker on one field, or ``None`` for a dynamic field."""
    markers = [item for item in model.model_fields[name].metadata if isinstance(item, SettingOwnership)]

    if len(markers) > 1:
        message = f"Settings field {name!r} has several ownership markers"
        raise ValueError(message)

    return markers[0] if markers else None


def keys_owned_by(model: type[BaseModel], ownership: SettingOwnership) -> frozenset[str]:
    return frozenset(name for name in model.model_fields if field_ownership(model, name) == ownership)


__all__ = ["Metadata", "Secret", "SettingOwnership", "Static", "field_ownership", "keys_owned_by"]
