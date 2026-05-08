"""Full-screen TUI for Valdoria SFT training using Textual.

Usage:
    app = SFTTrainingApp(cfg, model, tokenizer, trainer, ...)
    app.run()
"""

from __future__ import annotations

import json
import math
import time
import threading
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar, RichLog, Static
from textual.worker import Worker, WorkerState
from textual_plotext import PlotextPlot

from transformers import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


# ─── Shared helpers ─────────────────────────────────────────────────────

def _fmt(n: float | int | None, decimals: int = 2) -> str:
    if n is None:
        return "N/A"
    if isinstance(n, float):
        return f"{n:.{decimals}f}"
    return str(n)


def _fmt_time(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "--:--:--"
    return str(timedelta(seconds=int(seconds)))


def _fmt_lr(lr: float) -> str:
    if lr >= 1e-3:
        return f"{lr:.6f}"
    if lr >= 1e-6:
        return f"{lr:.3e}"
    return f"{lr:.2e}"


def load_authoring_categories(authoring_path: str) -> Counter:
    counter: Counter = Counter()
    path = Path(authoring_path)
    if not path.exists():
        return counter
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                tt = entry.get("task_type", "unknown")
                counter[tt] += 1
            except json.JSONDecodeError:
                pass
    return counter


# ─── Confirm Quit Screen ────────────────────────────────────────────────

class ConfirmQuitScreen(ModalScreen[bool]):
    """Modal dialog asking user to confirm stopping training."""

    CSS = """
    ConfirmQuitScreen {
        align: center middle;
    }
    #dialog {
        padding: 2 4;
        width: 40;
        height: auto;
        border: thick $primary;
        background: $surface;
    }
    #question {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    #buttons {
        align: center middle;
        height: auto;
    }
    Button {
        margin: 0 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Parar treino?", id="question")
            yield Label("Tem certeza que deseja interromper o treinamento?", id="question")
            with Vertical(id="buttons"):
                from textual.widgets import Button
                yield Button("Sim, parar", variant="error", id="confirm")
                yield Button("Continuar", variant="primary", id="cancel")

    def on_button_pressed(self, event) -> None:
        if event.button.id == "confirm":
            self.dismiss(True)
        else:
            self.dismiss(False)


# ─── Widget IDs ─────────────────────────────────────────────────────────

ID_DATASET_TABLE = "dataset-table"
ID_TRAINING_BOX = "training-box"
ID_EVAL_BOX = "eval-box"
ID_LOG = "log"
ID_STATUS = "status"


# ─── The TUI Application ────────────────────────────────────────────────

class SFTTrainingApp(App):
    """Full-screen TUI for monitoring Valdoria SFT training."""

    TITLE = "Valdoria SFT - Full Training"
    SUB_TITLE = "Qwen3.5-0.8B"

    CSS = """
    Screen {
        background: $surface;
    }

    Grid {
        grid-size: 2;
        grid-columns: 3fr 2fr;
        height: 1fr;
        grid-gutter: 1;
    }

    #left-column {
        height: 100%;
        overflow-y: auto;
    }

    #right-column {
        height: 100%;
        border-left: solid $primary;
        padding: 0 1;
    }

    #dataset-table {
        height: auto;
        max-height: 18;
        margin: 0 1;
    }

    #config-info, #dataset-info {
        height: auto;
        margin: 0 1;
    }

    #training-box {
        height: auto;
        margin: 0 1;
        border: solid $primary;
    }

    #eval-box {
        height: auto;
        margin: 0 1;
        border: solid $success;
    }

    #log {
        height: 1fr;
        margin: 0 1;
        border: solid $border;
        min-height: 5;
    }

    #status {
        height: 1;
        margin: 0 1;
        text-style: bold;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        padding: 0 1;
    }

    DataTable {
        height: auto;
    }

    .metric-line {
        height: 1;
        margin: 0 1;
    }

    PlotextPlot {
        height: 1fr;
        min-height: 10;
        margin: 1 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "confirm_quit", "Stop", priority=True),
        Binding("q", "confirm_quit", "Quit"),
        Binding("l", "clear_log", "Clear Log"),
    ]

    # Reactive attributes — setting these auto-updates widgets
    training_status = reactive("waiting")
    step_text = reactive("")
    loss_text = reactive("")
    lr_text = reactive("")
    gpu_progress = reactive(0.0)
    gpu_label_text = reactive("")
    speed_text = reactive("")
    eval_text = reactive("")

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: torch.nn.Module,
        tokenizer: Any,
        trainer: Trainer,
        train_file: str,
        val_file: str,
        authoring_categories: Optional[Counter] = None,
        max_steps: int = 0,
        max_seq_length: int = 768,
    ):
        super().__init__()
        self._cfg = cfg
        self._model = model
        self._tokenizer = tokenizer
        self._trainer = trainer
        self._train_file = train_file
        self._val_file = val_file
        self._authoring_cats = authoring_categories or Counter()
        self._max_steps = max_steps
        self._max_seq_length = max_seq_length

        # Runtime tracking
        self._start_time: float = 0.0
        self._last_log_time: float = 0.0
        self._last_step: int = 0
        self._training_started = False
        self._training_done = False
        self._train_worker: Optional[Worker] = None
        self._state_lock = threading.Lock()
        # Track recent step times for better ETA calculation
        self._recent_step_times: list[float] = []
        self._window_size: int = 20  # Use last 20 steps for ETA
        # Plot data storage
        self._loss_history: list[float] = []
        self._lr_history: list[float] = []
        self._gpu_history: list[float] = []
        self._step_history: list[int] = []
        self._max_plot_points: int = 100  # Limit points for performance

    # ─── Compose UI ───────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Grid():
            # Left Column - Existing content with scroll
            with VerticalScroll(id="left-column"):
                yield Static("Dataset", classes="section-title")
                yield DataTable(id=ID_DATASET_TABLE)
                yield Static(id="dataset-info")
                yield Static(id="config-info")

                yield Static("Training", classes="section-title")
                with Vertical(id=ID_TRAINING_BOX):
                    yield Static(id="step-info", classes="metric-line")
                    yield Static(id="loss-info", classes="metric-line")
                    yield Static(id="lr-info", classes="metric-line")
                    yield ProgressBar(id="gpu-bar", show_eta=False, show_percentage=False)
                    yield Static(id="gpu-label", classes="metric-line")
                    yield Static(id="speed-info", classes="metric-line")

                yield Static("Log", classes="section-title")
                yield RichLog(id=ID_LOG, highlight=True, markup=True, max_lines=100)

                yield Static("Evaluation", classes="section-title")
                with Vertical(id=ID_EVAL_BOX):
                    yield Static(id="eval-info")

            # Right Column - Plots
            with Vertical(id="right-column"):
                yield Static("Gráficos", classes="section-title")
                yield PlotextPlot(id="loss-plot")
                yield PlotextPlot(id="lr-plot")
                yield PlotextPlot(id="gpu-plot")

        yield Static(id=ID_STATUS)
        yield Footer()

    # ─── Mount ────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._populate_dataset_table()
        self._populate_dataset_info()
        self._populate_config_info()
        self._populate_model_info()
        self._init_plots()
        self.status("Pronto. Pressione qualquer tecla para iniciar o treino.")
        self.set_interval(5, self._update_elapsed)
        # Progress bar updates every 1s, even without logs
        self.set_interval(1, self._update_progress_from_state)

    def _populate_model_info(self) -> None:
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        pct = 100 * trainable / total
        model_name = self._cfg.get("model_name_or_path", "?")

        gpu_name = "N/A"
        gpu_vram = 0.0
        bf16_sup = False
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            gpu_name = props.name
            gpu_vram = props.total_memory / (1024**3)
            bf16_sup = torch.cuda.is_bf16_supported()

        self.query_one("#config-info", Static).update(
            f"Modelo: {model_name}  |  GPU: {gpu_name} ({gpu_vram:.1f} GB, bf16: {'sim' if bf16_sup else 'nao'})  "
            f"|  Parametros: {total / 1e6:.1f}M ({pct:.1f}% treinaveis)"
        )

    def _populate_dataset_table(self) -> None:
        table = self.query_one("#dataset-table", DataTable)
        table.add_columns("task_type", "Exemplos", "%", "Barra")
        total = self._authoring_cats.total()
        if total == 0:
            return
        max_cnt = max(self._authoring_cats.values())
        sorted_items = self._authoring_cats.most_common()
        for tt, cnt in sorted_items:
            pct = 100 * cnt / total
            bar_w = 20
            filled = int(bar_w * cnt / max_cnt)
            bar = "█" * filled + "░" * (bar_w - filled)
            table.add_row(tt, str(cnt), f"{pct:.1f}%", bar)

    def _populate_dataset_info(self) -> None:
        train_path = Path(self._train_file)
        val_path = Path(self._val_file)
        train_size = train_path.stat().st_size if train_path.exists() else 0
        val_size = val_path.stat().st_size if val_path.exists() else 0
        train_count = sum(1 for _ in open(train_path, encoding="utf-8")) if train_path.exists() else 0
        val_count = sum(1 for _ in open(val_path, encoding="utf-8")) if val_path.exists() else 0

        self.query_one("#dataset-info", Static).update(
            f"Train: {train_count} exemplos ({train_size / 1024:.0f} KB)  "
            f"|  Validation: {val_count} exemplos ({val_size / 1024:.0f} KB)"
        )

    def _populate_config_info(self) -> None:
        bs = int(self._cfg.get("per_device_train_batch_size", 1))
        ga = int(self._cfg.get("gradient_accumulation_steps", 16))
        eff_bs = bs * ga
        lr_v = float(self._cfg.get("learning_rate", 2e-5))
        sched = self._cfg.get("lr_scheduler_type", "cosine")
        optim = self._cfg.get("optim", "adamw_torch")
        epochs = float(self._cfg.get("num_train_epochs", 3))

        self.query_one("#config-info", Static).update(
            f"max_len: {self._max_seq_length}  |  batch: {bs} × {ga} = {eff_bs}  "
            f"|  lr: {lr_v}  |  scheduler: {sched}  |  optim: {optim}  "
            f"|  steps: {self._max_steps}  |  epochs: {epochs}"
        )

    # ─── Reactive watchers ────────────────────────────────────────────

    def watch_training_status(self, value: str) -> None:
        self.query_one("#status", Static).update(f"Status: {value}")

    def watch_step_text(self, value: str) -> None:
        self.query_one("#step-info", Static).update(value)

    def watch_loss_text(self, value: str) -> None:
        self.query_one("#loss-info", Static).update(value)

    def watch_lr_text(self, value: str) -> None:
        self.query_one("#lr-info", Static).update(value)

    def watch_gpu_progress(self, value: float) -> None:
        self.query_one("#gpu-bar", ProgressBar).progress = value

    def watch_gpu_label_text(self, value: str) -> None:
        self.query_one("#gpu-label", Static).update(value)

    def watch_speed_text(self, value: str) -> None:
        self.query_one("#speed-info", Static).update(value)

    def watch_eval_text(self, value: str) -> None:
        self.query_one("#eval-info", Static).update(value)

    # ─── Actions ──────────────────────────────────────────────────────

    def action_confirm_quit(self) -> None:
        if self._training_started and not self._training_done:
            self.push_screen(ConfirmQuitScreen(), self._on_quit_confirmed)
        else:
            self.exit()

    def _on_quit_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            self.status("Parando... aguardando fim do passo atual.")
            self.log_msg("[yellow]Parada solicitada pelo usuario[/]")
            self._trainer.args.max_steps = self._last_step + 1

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    # ─── Helpers ──────────────────────────────────────────────────────

    def log_msg(self, message: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.query_one("#log", RichLog).write(f"[dim]{now}[/] {message}")

    def status(self, msg: str) -> None:
        self.training_status = msg

    def _update_elapsed(self) -> None:
        """Periodic update so elapsed keeps ticking even without new logs."""
        if not self._training_started or self._training_done:
            return
        elapsed = time.time() - self._start_time
        if self.speed_text:
            parts = self.speed_text.rsplit("|", 1)
            if len(parts) == 2:
                self.speed_text = f"{parts[0]}|  Elapsed: {_fmt_time(elapsed)}"

    def _update_progress_from_state(self) -> None:
        """Update progress bar every second by reading trainer state directly."""
        if not self._training_started or self._training_done:
            return

        # Access trainer state safely with lock
        with self._state_lock:
            try:
                state = self._trainer.state
            except Exception:
                return

        if state is None:
            return

        step = state.global_step
        max_steps = state.max_steps or 1

        # Only update if we haven't shown this step yet (avoid spam when logs arrive)
        if step <= self._last_step:
            return

        step_pct = 100.0 * step / max_steps
        elapsed = time.time() - self._start_time

        # Show "processing" indicator until first real loss log
        bar_w = 20
        filled = int(bar_w * step / max_steps)
        bar = "█" * filled + "░" * (bar_w - filled)

        ep_str = ""
        if hasattr(state, 'epoch') and state.epoch is not None:
            ep_str = f"  Ep {_fmt(state.epoch, 2)}"

        self.step_text = f"Step {step}/{max_steps}  {bar}  {_fmt(step_pct, 1)}%  [processing]{ep_str}"

        # Show GPU usage
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / 1e9
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_pct = 100.0 * gpu_alloc / gpu_total if gpu_total > 0 else 0.0
            self.gpu_progress = gpu_pct
            self.gpu_label_text = f"GPU: {_fmt(gpu_alloc, 2)} GB / {_fmt(gpu_total, 1)} GB ({_fmt(gpu_pct, 1)}%)"

        # Update speed/eta with available info using recent step times
        if step > 0:
            if len(self._recent_step_times) > 0:
                avg_step_time = sum(self._recent_step_times) / len(self._recent_step_times)
            else:
                avg_step_time = elapsed / step
            remaining = avg_step_time * (max_steps - step)
            eta = datetime.now() + timedelta(seconds=remaining) if 0 < remaining < 86400 else None
            speed_part = f"Speed: {_fmt(avg_step_time, 1)} s/step"
            eta_part = f"ETA: {eta.strftime('%H:%M:%S') if eta else '--:--:--'}"
            elapsed_part = f"Elapsed: {_fmt_time(elapsed)}"
            self.speed_text = f"{speed_part}  |  {eta_part}  |  {elapsed_part}"

    def _init_plots(self) -> None:
        """Initialize the plots with titles and labels."""
        # Loss plot
        loss_widget = self.query_one("#loss-plot", PlotextPlot)
        loss_plt = loss_widget.plt
        loss_plt.title("Training Loss")
        loss_plt.xlabel("Steps")
        loss_plt.ylabel("Loss")
        loss_widget.refresh()

        # Learning Rate plot
        lr_widget = self.query_one("#lr-plot", PlotextPlot)
        lr_plt = lr_widget.plt
        lr_plt.title("Learning Rate")
        lr_plt.xlabel("Steps")
        lr_plt.ylabel("LR")
        lr_widget.refresh()

        # GPU plot
        gpu_widget = self.query_one("#gpu-plot", PlotextPlot)
        gpu_plt = gpu_widget.plt
        gpu_plt.title("GPU Memory %")
        gpu_plt.xlabel("Steps")
        gpu_plt.ylabel("%")
        gpu_widget.refresh()

    def _update_plots(self) -> None:
        """Update all plots with current data."""
        if not self._step_history:
            return

        # Loss plot
        loss_widget = self.query_one("#loss-plot", PlotextPlot)
        loss_plt = loss_widget.plt
        loss_plt.clear_data()
        loss_plt.plot(self._step_history, self._loss_history, color="red")
        loss_widget.refresh()

        # Learning Rate plot
        lr_widget = self.query_one("#lr-plot", PlotextPlot)
        lr_plt = lr_widget.plt
        lr_plt.clear_data()
        lr_plt.plot(self._step_history, self._lr_history, color="yellow")
        lr_widget.refresh()

        # GPU plot
        gpu_widget = self.query_one("#gpu-plot", PlotextPlot)
        gpu_plt = gpu_widget.plt
        gpu_plt.clear_data()
        gpu_plt.plot(self._step_history, self._gpu_history, color="cyan")
        gpu_widget.refresh()

    # ─── Start training ─────────────────────────────────────────────

    def on_key(self, event) -> None:
        if not self._training_started:
            self._start_training()

    def _start_training(self) -> None:
        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._last_step = 0
        self._training_started = True
        self.status("Treinando...")
        self.log_msg("[green]Treino iniciado[/]")
        self._run_train_worker()

    @work(thread=True, exit_on_error=False)
    def _run_train_worker(self) -> None:
        """Run training in a Textual worker thread."""
        callback = TUILogCallback(self)
        self._trainer.add_callback(callback)

        from transformers.trainer_callback import PrinterCallback
        self._trainer.callback_handler.callbacks = [
            c for c in self._trainer.callback_handler.callbacks
            if type(c).__name__ != "PrinterCallback"
        ]

        try:
            self._trainer.train(resume_from_checkpoint=False)
            # Schedule on main thread from worker thread
            self.call_later(self._on_training_done)
        except Exception as e:
            self.call_later(self.log_msg, f"[red]Erro no treino: {e}[/]")
            self.call_later(self.status, f"Erro: {e}")
            raise

    def _on_training_done(self) -> None:
        self._training_done = True
        self.log_msg("[green]Treino concluido![/]")
        self.status("Treino concluido. Pressione q ou Ctrl+C para sair.")

        try:
            metrics = self._trainer.evaluate()
            self._show_evaluation(metrics)
        except Exception as e:
            self.log_msg(f"[red]Erro na avaliacao: {e}[/]")

    def _show_evaluation(self, metrics: Dict[str, Any]) -> None:
        parts = []
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            try:
                perp = math.exp(eval_loss)
                parts.append(f"eval_loss: {eval_loss:.4f}  |  perplexity: {perp:.2f}")
            except (OverflowError, ValueError):
                parts.append(f"eval_loss: {eval_loss:.4f}  |  perplexity: INF")
        for k, v in metrics.items():
            if k == "eval_loss":
                continue
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        self.eval_text = "  |  ".join(parts)
        self.log_msg(f"[cyan]Avaliacao: {self.eval_text}[/]")

    def update_training_log(self, logs: Dict[str, Any], state: TrainerState) -> None:
        """Called from the callback (may be from worker thread)."""
        now = time.time()
        step = state.global_step
        max_steps = state.max_steps or 1
        step_pct = 100.0 * step / max_steps

        loss = logs.get("loss")
        try:
            perplexity = math.exp(float(loss)) if loss is not None else None
        except (OverflowError, ValueError):
            perplexity = float("inf")

        grad_norm = logs.get("grad_norm")
        learning_rate = logs.get("learning_rate")
        epoch = logs.get("epoch")

        gpu_alloc = 0.0
        gpu_total = 0.0
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / 1e9
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Track step time for ETA calculation
        if step > self._last_step and self._last_log_time > 0:
            step_time = (now - self._last_log_time) / (step - self._last_step)
            # Store recent step times for better ETA
            self._recent_step_times.append(step_time)
            if len(self._recent_step_times) > self._window_size:
                self._recent_step_times.pop(0)
        else:
            step_time = 0.0

        self._last_log_time = now
        self._last_step = step

        # Calculate ETA using recent step times (better than using total average)
        elapsed = now - self._start_time
        if len(self._recent_step_times) > 0:
            avg_step_time = sum(self._recent_step_times) / len(self._recent_step_times)
            remaining = avg_step_time * (max_steps - step)
        elif step > 0:
            avg_step_time = elapsed / step
            remaining = avg_step_time * (max_steps - step)
        else:
            remaining = 0.0

        eta = datetime.now() + timedelta(seconds=remaining) if remaining > 0 and remaining < 86400 else None

        bar_w = 20
        filled = int(bar_w * step / max_steps)
        bar = "█" * filled + "░" * (bar_w - filled)

        ep_str = f"Ep {_fmt(epoch, 2)}" if epoch is not None else ""
        indicator = "[processing]" if loss is None else ""
        self.step_text = f"Step {step}/{max_steps}  {bar}  {_fmt(step_pct, 1)}%  {indicator}{ep_str}"

        loss_part = f"Loss: {_fmt(loss, 4) if loss else 'N/A'}"
        perp_part = f"Perplexity: {_fmt(perplexity, 2) if perplexity != float('inf') else 'INF'}"
        self.loss_text = f"{loss_part}  |  {perp_part}"

        lr_part = f"LR: {_fmt_lr(learning_rate) if learning_rate else 'N/A'}"
        grad_part = f"Grad Norm: {_fmt(grad_norm, 2) if grad_norm else 'N/A'}"
        self.lr_text = f"{lr_part}  |  {grad_part}"

        gpu_pct = 100.0 * gpu_alloc / gpu_total if gpu_total > 0 else 0.0
        self.gpu_progress = gpu_pct
        self.gpu_label_text = f"GPU: {_fmt(gpu_alloc, 2)} GB / {_fmt(gpu_total, 1)} GB ({_fmt(gpu_pct, 1)}%)"

        # Collect data for plots
        if loss is not None:
            self._loss_history.append(float(loss))
        if learning_rate is not None:
            self._lr_history.append(float(learning_rate))
        self._gpu_history.append(gpu_pct)
        self._step_history.append(step)

        # Limit history size for performance
        if len(self._step_history) > self._max_plot_points:
            self._loss_history = self._loss_history[-self._max_plot_points:]
            self._lr_history = self._lr_history[-self._max_plot_points:]
            self._gpu_history = self._gpu_history[-self._max_plot_points:]
            self._step_history = self._step_history[-self._max_plot_points:]

        speed_part = f"Speed: {_fmt(step_time, 1)} s/step" if step_time > 0 else "Speed: N/A"
        eta_part = f"ETA: {eta.strftime('%H:%M:%S') if eta else '--:--:--'}"
        elapsed_part = f"Elapsed: {_fmt_time(elapsed)}"
        self.speed_text = f"{speed_part}  |  {eta_part}  |  {elapsed_part}"

        log_parts = [f"Step {step}"]
        if loss is not None:
            log_parts.append(f"loss={_fmt(loss, 4)}")
        if learning_rate is not None:
            log_parts.append(f"lr={_fmt_lr(learning_rate)}")
        if grad_norm is not None:
            log_parts.append(f"grad={_fmt(grad_norm, 2)}")
        if perplexity is not None and perplexity != float("inf"):
            log_parts.append(f"perp={_fmt(perplexity, 2)}")
        self.log_msg(" | ".join(log_parts))

        # Update plots every 5 steps to avoid overloading
        if step % 5 == 0:
            self._update_plots()

    def update_evaluation(self, metrics: Dict[str, Any]) -> None:
        # This is called from the callback thread via call_from_thread,
        # so we can safely schedule on main thread
        self.call_later(self._show_evaluation, metrics)


# ─── Trainer Callback ─────────────────────────────────────────────────

class TUILogCallback(TrainerCallback):
    """Callback that forwards training logs to the TUI application."""

    def __init__(self, app: SFTTrainingApp):
        self.app = app
        self._eval_seen: set[int] = set()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get("logs", {})
        if not logs:
            return

        if "eval_loss" in logs:
            if state.global_step not in self._eval_seen:
                self._eval_seen.add(state.global_step)
                # Called from HF Trainer thread, need to schedule on main Textual thread
                self.app.call_from_thread(self.app.update_evaluation, logs)
        elif "loss" in logs:
            # Called from HF Trainer thread, need to schedule on main Textual thread
            self.app.call_from_thread(self.app.update_training_log, logs, state)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
