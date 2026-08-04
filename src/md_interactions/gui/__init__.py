"""Local graphical interface with a 3D structure viewer."""

from __future__ import annotations

from .server import GuiState, serve

__all__ = ["serve", "GuiState"]
