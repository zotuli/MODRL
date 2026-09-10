"""Environment factory and simple benchmark problems.

The training loop only requires a problem object with bounds, objective
dimension, and an ``obj`` method. This makes it possible to test the same
Gray-code policy on standard multi-objective benchmarks before using the
expensive parallel-robot objective.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProblemFactory = Callable[[int], Any]


def _osyczka_kundu(bits: int) -> Any:
    from .MooEnv_gray import osyczka_kundu

    problem = osyczka_kundu()
    problem.x_bin_num = bits
    return problem


def _zdt2(bits: int) -> Any:
    from .MooEnv_gray import zdt2

    return zdt2(bits)


def _zdt6(bits: int) -> Any:
    from .MooEnv_gray import zdt6

    return zdt6(bits)


def _kursawe(bits: int) -> Any:
    from .MooEnv_gray import kursawe

    return kursawe(bits)


PROBLEM_FACTORIES: dict[str, ProblemFactory] = {
    "osyczka_kundu": _osyczka_kundu,
    "zdt2": _zdt2,
    "zdt6": _zdt6,
    "kursawe": _kursawe,
}


def register_problem(name: str, factory: ProblemFactory) -> None:
    """Register a custom simple problem factory for programmatic use."""

    if not name or name == "parallel_robot":
        raise ValueError("Use a non-empty name other than 'parallel_robot'.")
    PROBLEM_FACTORIES[name] = factory


def build_environment(
    environment: str,
    idx: int,
    bits: int,
    batch_size: int,
    workspace_method: str = "PRBD",
) -> tuple[Any, Any]:
    """Build either the robot environment or a lightweight benchmark.

    A custom problem factory should return an object exposing ``obj_dim``,
    ``x_dim``, ``x_low``, ``x_up``, ``inital_state``, and ``obj(x)``. Its
    objectives are interpreted as minimization objectives by the current
    Pareto-reward implementation.
    """

    method = workspace_method.upper()
    if environment == "parallel_robot":
        from .MooEnv_gray import MooEnv, MooSCH, MooTestSettingA
        from . import function_revised as robot_backend
        from .function_revised import generRefPoints

        reference_points = generRefPoints(idx)
        problem = MooSCH(
            reference_points,
            bits,
            idx,
            ws_method=method,
            backend=robot_backend,
        )
    else:
        if method != "PRBD":
            raise ValueError("--workspace-method is only valid for parallel_robot.")
        try:
            factory = PROBLEM_FACTORIES[environment]
        except KeyError as exc:
            available = ", ".join(sorted(PROBLEM_FACTORIES))
            raise ValueError(
                f"Unknown environment '{environment}'. Available simple problems: {available}."
            ) from exc

        from .MooEnv_gray import MooEnv, MooTestSettingA

        problem = factory(bits)

    setting = MooTestSettingA(moo_pro=problem, batch_size=batch_size)
    return MooEnv(setting), problem
