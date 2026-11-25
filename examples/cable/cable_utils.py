# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared solver selection for the Newton cable examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import newton
import newton.examples
import style3d_sim as s3d
from style3d_sim.cable_material import register_cable_attributes


@dataclass(frozen=True)
class CableSolverProfile:
    """SynReal work budget for one nonlinear solver in one example."""

    nonlinear_iterations: int
    linear_iterations: int
    affine_body_block_preconditioner: bool = False
    affine_body_contact_block_preconditioner: bool = True
    cable_linear_beta: float | None = None


# Scene-specific work budgets and remaining alignment limitations are documented
# in docs/cable_validation.md. Each solver has its own entry so tuning PD never
# changes Newton implicitly.
CABLE_SOLVER_PROFILES: dict[str, dict[str, CableSolverProfile]] = {
    "example_cable_bundle_hysteresis": {
        "synreal": CableSolverProfile(2, 5, True, False), "pd": CableSolverProfile(2, 5, True, False),
    },
    "example_cable_cross_slide_table": {
        "synreal": CableSolverProfile(2, 4, True), "pd": CableSolverProfile(2, 4, True),
    },
    "example_cable_pile": {
        "synreal": CableSolverProfile(1, 2), "pd": CableSolverProfile(1, 2),
    },
    "example_cable_coil_tie": {
        "synreal": CableSolverProfile(2, 5, True, False), "pd": CableSolverProfile(2, 5, True, False),
    },
    "example_cable_plectoneme": {
        "synreal": CableSolverProfile(2, 5, True), "pd": CableSolverProfile(2, 5, True),
    },
    "example_cable_twist": {
        "synreal": CableSolverProfile(1, 2), "pd": CableSolverProfile(1, 2),
    },
    "example_cable_y_junction": {
        "synreal": CableSolverProfile(1, 5), "pd": CableSolverProfile(1, 5),
    },
    "example_franka_cable_ik_pick_place": {
        "synreal": CableSolverProfile(2, 9, True, False, cable_linear_beta=1000.0),
        "pd": CableSolverProfile(2, 9, True, cable_linear_beta=1000.0),
    },
    "example_mujoco_franka_vbd_cable_admm_solver": {
        "synreal": CableSolverProfile(2, 10, True), "pd": CableSolverProfile(2, 10, True),
    },
    "example_cable_bend_stiffness": {
        "synreal": CableSolverProfile(1, 5), "pd": CableSolverProfile(1, 5),
    },
    "example_cable_bend_twist_analytic": {
        "synreal": CableSolverProfile(1, 1), "pd": CableSolverProfile(1, 1),
    },
    "example_cable_dahl_hysteresis": {
        "synreal": CableSolverProfile(1, 4), "pd": CableSolverProfile(1, 4),
    },
    "example_cable_michell_threshold": {
        "synreal": CableSolverProfile(1, 5, True), "pd": CableSolverProfile(1, 5, True),
    },
    "example_cable_torsion_material_mapping": {
        "synreal": CableSolverProfile(1, 1), "pd": CableSolverProfile(1, 1),
    },
    "example_cable_twist_buckling": {
        "synreal": CableSolverProfile(1, 5, True), "pd": CableSolverProfile(1, 5, True),
    },
    "example_cable_twist_transfer": {
        "synreal": CableSolverProfile(1, 1), "pd": CableSolverProfile(1, 1),
    },
}


def create_cable_parser(example_name: str) -> argparse.ArgumentParser:
    """Attach an explicit example profile key; CLI iteration overrides take precedence.

    Pass ``Path(__file__).stem`` from the example entry point. Profile selection
    changes solver work budgets only; it does not change authored scene settings.
    """
    if example_name not in CABLE_SOLVER_PROFILES:
        raise ValueError(f"Unknown cable example profile: {example_name}")
    parser = newton.examples.create_parser()
    parser.set_defaults(cable_example_name=example_name)
    parser.add_argument(
        "--solver", choices=("synreal", "pd", "vbd"), default="synreal",
        help="Cable solver: SynReal Newton (default), SynReal PD, or the upstream Newton VBD reference.",
    )
    parser.add_argument(
        "--nonlinear-iterations", type=int, default=None,
        help="Override the example profile's SynReal nonlinear iterations per substep.",
    )
    parser.add_argument(
        "--linear-iterations", type=int, default=None,
        help="Override the example profile's SynReal PCG iterations per nonlinear iteration.",
    )
    parser.add_argument(
        "--body-block-preconditioner", action=argparse.BooleanOptionalAction, default=None,
        help="Use the optional full affine-body PCG block preconditioner.",
    )
    parser.add_argument(
        "--inertia-mode",
        choices=("explicit-euler", "implicit-euler", "lie-group-midpoint", "midpoint-cayley", "vbd-semi-implicit"),
        default=None, help="Override the SynReal rigid-body inertia integrator.",
    )
    parser.add_argument(
        "--contact-block-preconditioner", action=argparse.BooleanOptionalAction, default=None,
        help="Include hard-contact cross terms in affine-body PCG blocks.",
    )
    parser.add_argument(
        "--cable-linear-beta", type=float, default=None,
        help="Override the stretch/shear penalty ramp per SynReal nonlinear iteration.",
    )
    return parser


def create_cable_builder(args: Any = None, **kwargs: Any) -> newton.ModelBuilder:
    if getattr(args, "solver", "synreal") == "vbd":
        return newton.ModelBuilder(**kwargs)
    builder = s3d.ModelBuilder(**kwargs)
    # These examples compare the same authored material/contact model across
    # solvers. The general SynReal builder's stiffer contact defaults would alter it.
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig()
    return builder


def create_cable_solver(args: Any, model: newton.Model, **vbd_options: Any):
    """Apply the selected example/solver profile, then explicit CLI overrides.

    A programmatic caller without a parser profile retains the general SynReal
    configuration defaults. VBD always uses the original scene's solver options.
    """
    if getattr(args, "solver", "synreal") == "vbd":
        return newton.solvers.SolverVBD(model, **vbd_options)
    solver = s3d.NewtonSolver(model)
    inertia_mode = getattr(args, "inertia_mode", None)
    if inertia_mode is not None:
        solver.config.inertia_mode = getattr(s3d.InertiaMode, inertia_mode.replace("-", "_").upper())
    # Preserve the scene's VBD contact law, including defaults omitted by the
    # example. C0 stabilization and history change the mechanics of gripping.
    contact_alpha = vbd_options.get("rigid_avbd_contact_alpha")
    if contact_alpha is None:
        contact_alpha = vbd_options.get("rigid_avbd_alpha", 0.95)
    contact_beta = vbd_options.get("rigid_avbd_linear_beta")
    if contact_beta is None:
        contact_beta = vbd_options.get("rigid_avbd_beta", 0.0)
    contact_options = {
        "rigid_contact_friction_epsilon": vbd_options.get("friction_epsilon", 0.01),
        "rigid_contact_hard": vbd_options.get("rigid_contact_hard", True),
        "rigid_contact_history": vbd_options.get("rigid_contact_history", False),
        "rigid_contact_alpha": contact_alpha,
        "rigid_contact_beta": contact_beta,
        "rigid_contact_gamma": vbd_options.get("rigid_avbd_gamma", 0.999),
        "rigid_contact_k_start": vbd_options.get("rigid_contact_k_start", 100.0),
    }
    for name, value in contact_options.items():
        setattr(solver.config, name, value)
    angular_beta = vbd_options.get("rigid_avbd_angular_beta")
    if angular_beta is None:
        angular_beta = vbd_options.get("rigid_avbd_beta", 0.0)
    # AVBD ramps soft cable penalties too; authored stiffnesses remain their caps.
    cable_options = {
        "cable_linear_beta": contact_beta,
        "cable_angular_beta": angular_beta,
        "cable_penalty_gamma": vbd_options.get("rigid_avbd_gamma", 0.999),
        "cable_linear_k_start": vbd_options.get("rigid_joint_linear_k_start", 100.0),
        "cable_angular_k_start": vbd_options.get("rigid_joint_angular_k_start", 10.0),
    }
    for name, value in cable_options.items():
        setattr(solver.config, name, value)
    solver.config.nonlinear_solver = (
        s3d.NonlinearSolver.PD_NEWTON if getattr(args, "solver", "synreal") == "pd" else s3d.NonlinearSolver.NEWTON
    )
    nonlinear_iterations = getattr(args, "nonlinear_iterations", None)
    linear_iterations = getattr(args, "linear_iterations", None)
    example_name = getattr(args, "cable_example_name", None)
    body_block = getattr(args, "body_block_preconditioner", None)
    contact_block = getattr(args, "contact_block_preconditioner", None)
    linear_beta = getattr(args, "cable_linear_beta", None)
    if example_name is not None:
        profile = CABLE_SOLVER_PROFILES[example_name][getattr(args, "solver", "synreal")]
        if nonlinear_iterations is None:
            nonlinear_iterations = profile.nonlinear_iterations
        if linear_iterations is None:
            linear_iterations = profile.linear_iterations
        if body_block is None:
            body_block = profile.affine_body_block_preconditioner
        if contact_block is None:
            contact_block = profile.affine_body_contact_block_preconditioner
        if linear_beta is None:
            linear_beta = profile.cable_linear_beta
    if body_block is not None:
        solver.config.affine_body_block_preconditioner = body_block
    if contact_block is not None:
        if hasattr(solver.config, "affine_body_contact_block_preconditioner"):
            solver.config.affine_body_contact_block_preconditioner = contact_block
        elif not contact_block:
            raise RuntimeError("Rebuild the native extension to disable contact block preconditioning.")
    if linear_beta is not None:
        import math

        if not math.isfinite(linear_beta) or linear_beta < 0.0:
            raise ValueError("--cable-linear-beta must be finite and nonnegative")
        # Match the reference's ramp rate at a lower nonlinear work budget;
        # this changes the optimization schedule, not authored stiffness caps.
        solver.config.cable_linear_beta = linear_beta
    if nonlinear_iterations is not None:
        if nonlinear_iterations < 0:
            raise ValueError("--nonlinear-iterations must be nonnegative")
        solver.config.nonlinear_iterations = nonlinear_iterations
    if linear_iterations is not None:
        if linear_iterations < 1:
            raise ValueError("--linear-iterations must be positive")
        solver.config.linear_iterations = linear_iterations
    return solver


def update_cable_contact_history(solver: Any, enabled: bool) -> None:
    # Match indices describe a new collision query, so substeps that reuse the
    # same contact buffer must retain the current multipliers without remapping.
    if hasattr(solver, "set_rigid_history_update"):
        solver.set_rigid_history_update(enabled)


def prepare_cable_contacts(
    solver: Any, contacts: Any, *, proxy_pairs: tuple[tuple[str, str], ...] = (),
) -> None:
    """Prepare actual collision buffers before coupled CUDA graph capture."""
    prepare = getattr(solver, "prepare_contacts", None)
    if prepare is not None:
        prepare(contacts)
    entry_names = getattr(solver, "entry_names", None)
    if entry_names is None:
        return
    # The coupled prepare hook allocates filtered buffers but does not forward
    # their capacity to child solvers. Filtering preserves the parent capacity.
    for name in entry_names():
        child_prepare = getattr(solver.solver(name), "prepare_contacts", None)
        if child_prepare is not None:
            child_prepare(contacts)
    # Proxy destinations consume a separate pipeline whose capacity can exceed
    # the parent buffer. Reserve only the explicitly configured proxy directions.
    for source, destination in proxy_pairs:
        proxy_contacts = solver.get_proxy_contacts(source, destination)
        child_prepare = getattr(solver.solver(destination), "prepare_contacts", None)
        if child_prepare is not None and proxy_contacts is not None:
            child_prepare(proxy_contacts)
