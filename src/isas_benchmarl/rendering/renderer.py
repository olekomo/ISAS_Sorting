"""Matplotlib renderer isolated from the training hot path."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ..config import ISASConfig
from ..enums import ActorStatus
from .snapshot import RenderSnapshot


class MatplotlibRenderer:
    """Render a snapshot without retaining simulation tensors or GPU state."""

    def __init__(self, config: ISASConfig, dpi: int = 120):
        self.config = config
        self.dpi = dpi
        self._human_figure = None

    def render(self, snapshot: RenderSnapshot) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        width_inches = max(7.0, snapshot.array_end / 20.0)
        height_inches = max(4.0, snapshot.area_width / 20.0)
        fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=self.dpi)
        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                snapshot.array_end,
                snapshot.area_width,
                fill=False,
                linewidth=1.5,
            )
        )

        status_colors = {
            int(ActorStatus.READY): "lightgray",
            int(ActorStatus.ACTIVATE): "gold",
            int(ActorStatus.UP): "orange",
            int(ActorStatus.HIT): "crimson",
            int(ActorStatus.DOWN): "darkorange",
            int(ActorStatus.RESET): "slategray",
        }
        for index, ((x, y), status, delay) in enumerate(
            zip(
                snapshot.actor_positions,
                snapshot.actor_status,
                snapshot.pending_delay,
            )
        ):
            ax.add_patch(
                Rectangle(
                    (x - snapshot.actor_length / 2.0, y - snapshot.actor_width / 2.0),
                    snapshot.actor_length,
                    snapshot.actor_width,
                    facecolor=status_colors[int(status)],
                    edgecolor="black",
                    alpha=0.8,
                )
            )
            label = str(index)
            if delay >= 0:
                label += f"\n+{int(delay)}"
            ax.text(x, y, label, ha="center", va="center", fontsize=6)

        for state, pclass, spawned, active, outcome in zip(
            snapshot.particle_state,
            snapshot.particle_class,
            snapshot.particle_spawned,
            snapshot.particle_active,
            snapshot.particle_outcome,
        ):
            if not spawned:
                continue
            x, _, y, _ = state
            if active:
                face = "red" if int(pclass) == 1 else "royalblue"
                alpha = 0.9
            else:
                face = {1: "green", 2: "orange", 3: "gray"}.get(
                    int(outcome), "gray"
                )
                alpha = 0.35
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=snapshot.particle_radius,
                    facecolor=face,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=alpha,
                )
            )

        ax.set_xlim(-max(5.0, snapshot.particle_radius * 2), snapshot.array_end + 5.0)
        ax.set_ylim(0.0, snapshot.area_width)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"ISAS sorting – physics step {snapshot.physics_step}")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=self.dpi)
        plt.close(fig)
        buffer.seek(0)
        frame = np.asarray(Image.open(buffer).convert("RGB")).copy()
        buffer.close()
        return frame

    def show(self, frame: np.ndarray) -> None:
        import matplotlib.pyplot as plt

        if self._human_figure is None:
            self._human_figure, axis = plt.subplots()
            self._human_axis = axis
            self._human_image = axis.imshow(frame)
            axis.axis("off")
        else:
            self._human_image.set_data(frame)
        self._human_figure.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        if self._human_figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._human_figure)
            self._human_figure = None
