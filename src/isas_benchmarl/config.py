"""Typed configuration for the vectorized ISAS sorting simulator.

The public field names intentionally mirror the legacy ``EnvConfig`` wherever
possible. Only ``max_fire_delay_steps`` and ``max_decision_steps`` are new and
required by the delayed-discrete action interface and a bounded TorchRL rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, Mapping

import yaml


def _get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value {value!r}.")
    return bool(value)


def _required(source: Any, key: str) -> Any:
    value = _get(source, key, None)
    if value is None:
        raise ValueError(f"Missing required configuration value: {key}")
    return value


@dataclass(frozen=True)
class RewardCostConfig:
    accept_cost: float
    reject_cost: float
    disturbed_cost: float

    @classmethod
    def from_source(cls, source: Any) -> "RewardCostConfig":
        nested = _get(source, "reward_costs", None)
        values = nested if nested is not None else source
        return cls(
            accept_cost=float(_required(values, "accept_cost")),
            reject_cost=float(_required(values, "reject_cost")),
            disturbed_cost=float(_required(values, "disturbed_cost")),
        )


@dataclass(frozen=True)
class GridConfig:
    name: str
    rows: int
    cols: int
    row_spacing: float
    col_spacing: float
    row_offset: float
    col_offsets: tuple[float, ...]
    actor_positions: tuple[tuple[float, float], ...]
    array_end: float
    area_width: float
    default_birth_rate: float

    @property
    def num_actors(self) -> int:
        return len(self.actor_positions)

    @classmethod
    def from_source(cls, name: str, source: Any) -> "GridConfig":
        # Existing EnvConfig already exposes actor_positions. Plain mappings
        # usually expose the compact row/column representation.
        existing_positions = _get(source, "actor_positions", None)
        rows = int(_required(source, "rows"))
        cols = int(_required(source, "cols"))
        row_spacing = float(_required(source, "row_spacing"))
        col_spacing = float(_required(source, "col_spacing"))
        row_offset = float(_required(source, "row_offset"))
        col_offsets = tuple(float(v) for v in _required(source, "col_offsets"))

        if len(col_offsets) != rows:
            raise ValueError(
                f"grid.col_offsets has length {len(col_offsets)}, expected {rows}."
            )

        if existing_positions is None:
            positions: list[tuple[float, float]] = []
            for row in range(rows):
                x = row * row_spacing + row_offset
                for col in range(cols):
                    y = col * col_spacing + col_offsets[row]
                    positions.append((float(x), float(y)))
            positions.sort(key=lambda p: (p[0], p[1]))
            actor_positions = tuple(positions)
        else:
            actor_positions = tuple(
                (float(position[0]), float(position[1]))
                for position in existing_positions
            )

        return cls(
            name=name,
            rows=rows,
            cols=cols,
            row_spacing=row_spacing,
            col_spacing=col_spacing,
            row_offset=row_offset,
            col_offsets=col_offsets,
            actor_positions=actor_positions,
            array_end=float(_required(source, "array_end")),
            area_width=float(_required(source, "area_width")),
            default_birth_rate=float(_required(source, "default_birth_rate")),
        )


@dataclass(frozen=True)
class ISASConfig:
    # Legacy-compatible public fields.
    grid_size: str
    T: float
    obs_num_particles: int
    num_particles: int
    seed: int
    grid: GridConfig
    area_width: float

    t_activate: float
    t_up: float
    t_hit: float
    t_down: float
    t_reset: float
    actor_length: float
    actor_width: float

    v0_x: float
    simulation_noise: float
    particle_spawn_x: float
    particle_spawn_y_fraction: float
    particle_spawn_x_stddev: float
    particle_spawn_y_stddev_factor: float
    particle_v0_x_stddev: float
    particle_v0_y_mean: float
    particle_v0_y_stddev: float
    particle_birth_rate_stddev: float
    particle_class_labels: tuple[int, ...]
    particle_radius: float
    particle_velocity_y_min: float
    particle_velocity_y_max: float

    dist_area_len: float
    dist_area_width: float
    dist_area_off_x: float
    dist_area_off_y: float

    frame_factor: int
    render_mode: str | None
    save_animation: bool
    animation_dir: str
    agent_centric_obs: bool
    continuous_actor_states: bool
    normalize_rewards: bool
    time_out: int
    reward_costs: RewardCostConfig

    # New fields.
    max_fire_delay_steps: int = 8
    max_decision_steps: int = 1000
    y_distance_weight: float = 100.0

    @property
    def num_actors(self) -> int:
        return self.grid.num_actors

    @property
    def num_actions(self) -> int:
        # 0=no-op, 1=activate now, d+1=activate after d physics steps.
        return self.max_fire_delay_steps + 2

    @property
    def cycle_time(self) -> float:
        return self.t_activate + self.t_up + self.t_hit + self.t_down + self.t_reset

    @property
    def decision_period(self) -> float:
        return self.T * self.frame_factor

    def duration_to_steps(self, value: float, name: str) -> int:
        raw = value / self.T
        rounded = int(round(raw))
        if not isclose(raw, rounded, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"{name}={value} must be an integer multiple of T={self.T}; "
                f"got {raw} physics steps."
            )
        return rounded

    @property
    def actor_phase_steps(self) -> tuple[int, int, int, int, int]:
        return (
            self.duration_to_steps(self.t_activate, "t_activate"),
            self.duration_to_steps(self.t_up, "t_up"),
            self.duration_to_steps(self.t_hit, "t_hit"),
            self.duration_to_steps(self.t_down, "t_down"),
            self.duration_to_steps(self.t_reset, "t_reset"),
        )

    @classmethod
    def from_source(cls, source: Any) -> "ISASConfig":
        """Load from a plain mapping or the existing project ``EnvConfig``.

        A BenchMARL task mapping may wrap environment values below
        ``config_dict``. That wrapper is unwrapped here.
        """
        if isinstance(source, ISASConfig):
            return source
        wrapped = _get(source, "config_dict", None)
        if wrapped is not None:
            source = wrapped

        if isinstance(source, Mapping) and "profile_name" in source:
            try:
                from src.core.configs.experiment_config import (
                    load_experiment_profile,
                )
            except ModuleNotFoundError as error:
                raise ValueError(
                    "profile_name requires the legacy project's "
                    "load_experiment_profile module. Pass the resolved EnvConfig "
                    "or a complete mapping when using this standalone package."
                ) from error
            profile = load_experiment_profile(source["profile_name"])
            resolved = dict(profile.environment_raw)
            if "params" in profile.reward:
                resolved.update(profile.reward["params"])
            source = resolved

        grid_size = str(_required(source, "grid_size"))
        grid_source = _required(source, "grid")
        grid = GridConfig.from_source(grid_size, grid_source)
        reward_costs = RewardCostConfig.from_source(source)

        config = cls(
            grid_size=grid_size,
            T=float(_required(source, "T")),
            obs_num_particles=int(_required(source, "obs_num_particles")),
            num_particles=int(_required(source, "num_particles")),
            seed=int(_get(source, "seed", 0)),
            grid=grid,
            area_width=float(_get(source, "area_width", grid.area_width)),
            t_activate=float(_required(source, "t_activate")),
            t_up=float(_required(source, "t_up")),
            t_hit=float(_required(source, "t_hit")),
            t_down=float(_required(source, "t_down")),
            t_reset=float(_required(source, "t_reset")),
            actor_length=float(_required(source, "actor_length")),
            actor_width=float(_required(source, "actor_width")),
            v0_x=float(_required(source, "v0_x")),
            simulation_noise=float(_required(source, "simulation_noise")),
            particle_spawn_x=float(_required(source, "particle_spawn_x")),
            particle_spawn_y_fraction=float(
                _required(source, "particle_spawn_y_fraction")
            ),
            particle_spawn_x_stddev=float(
                _required(source, "particle_spawn_x_stddev")
            ),
            particle_spawn_y_stddev_factor=float(
                _required(source, "particle_spawn_y_stddev_factor")
            ),
            particle_v0_x_stddev=float(
                _required(source, "particle_v0_x_stddev")
            ),
            particle_v0_y_mean=float(_required(source, "particle_v0_y_mean")),
            particle_v0_y_stddev=float(
                _required(source, "particle_v0_y_stddev")
            ),
            particle_birth_rate_stddev=float(
                _required(source, "particle_birth_rate_stddev")
            ),
            particle_class_labels=tuple(
                int(v) for v in _required(source, "particle_class_labels")
            ),
            particle_radius=float(_required(source, "particle_radius")),
            particle_velocity_y_min=float(
                _required(source, "particle_velocity_y_min")
            ),
            particle_velocity_y_max=float(
                _required(source, "particle_velocity_y_max")
            ),
            dist_area_len=float(_required(source, "dist_area_len")),
            dist_area_width=float(_required(source, "dist_area_width")),
            dist_area_off_x=float(_required(source, "dist_area_off_x")),
            dist_area_off_y=float(_required(source, "dist_area_off_y")),
            frame_factor=int(_required(source, "frame_factor")),
            render_mode=_get(source, "render_mode", None),
            save_animation=_as_bool(_get(source, "save_animation", False)),
            animation_dir=str(_get(source, "animation_dir", "./vis_results/")),
            agent_centric_obs=_as_bool(_get(source, "agent_centric_obs", True)),
            continuous_actor_states=_as_bool(
                _get(source, "continuous_actor_states", True)
            ),
            normalize_rewards=_as_bool(_get(source, "normalize_rewards", False)),
            time_out=int(_get(source, "time_out", 30)),
            reward_costs=reward_costs,
            max_fire_delay_steps=int(_get(source, "max_fire_delay_steps", 8)),
            max_decision_steps=int(_get(source, "max_decision_steps", 1000)),
            y_distance_weight=float(_get(source, "y_distance_weight", 100.0)),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ISASConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            values = yaml.safe_load(file)
        return cls.from_source(values)

    def validate(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.frame_factor <= 0:
            raise ValueError("frame_factor must be positive.")
        if self.num_particles <= 0 or self.obs_num_particles <= 0:
            raise ValueError("num_particles and obs_num_particles must be positive.")
        if self.max_fire_delay_steps < 0:
            raise ValueError("max_fire_delay_steps must be non-negative.")
        if self.max_decision_steps <= 0:
            raise ValueError("max_decision_steps must be positive.")
        if self.particle_radius <= 0:
            raise ValueError("particle_radius must be positive.")
        if 2 * self.particle_radius >= self.area_width:
            raise ValueError("particle_radius leaves no valid y spawn interval.")
        if self.v0_x <= 0:
            raise ValueError("v0_x must be positive.")
        if self.grid.array_end <= max(p[0] for p in self.grid.actor_positions):
            raise ValueError("grid.array_end must be behind every actor.")
        labels = set(self.particle_class_labels)
        if not labels.issubset({0, 1}) or labels != {0, 1}:
            raise ValueError(
                "particle_class_labels must contain both class 0 (keep) and "
                "class 1 (eject)."
            )
        _ = self.actor_phase_steps


def load_config(source: Any) -> ISASConfig:
    """Public compatibility helper."""
    return ISASConfig.from_source(source)
