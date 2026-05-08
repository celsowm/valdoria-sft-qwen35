"""Display utilities for Valdoria SFT training.

Re-exports the TUI application and shared helper functions.
Use `run_tui(...)` to launch the full-screen training monitor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .tui_app import (
    SFTTrainingApp,
    load_authoring_categories,
)


def run_tui(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    trainer: Any,
    train_file: str,
    val_file: str,
    authoring_categories: Optional[object] = None,
    max_steps: int = 0,
    max_seq_length: int = 768,
) -> None:
    """Launch the full-screen TUI and block until training completes or user quits."""
    app = SFTTrainingApp(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        train_file=train_file,
        val_file=val_file,
        authoring_categories=authoring_categories,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
    )
    app.run()
