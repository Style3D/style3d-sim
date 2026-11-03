# SPDX-FileCopyrightText: Copyright (c) 2026 Zhejiang Linctex Digital Technology Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional Newton Physics adapter for the native Style3D Sim solver."""

from __future__ import annotations

import getpass
from importlib import import_module
from itertools import combinations
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from ._core import Config, Contact, Control, Model, Solver, Span, State, is_logged_in, login
from .joint_drive_effort import evaluate_joint_drive_effort
from .rigid_body_dynamics import (
    NONE,
    VBD_SEMI_IMPLICIT,
    integrate_rigid_body,
    integrate_vbd_semi_implicit,
    recover_rigid_body_end_velocity,
)

if TYPE_CHECKING:
    from .newton_model_builder import ModelBuilder

__all__ = ["ModelBuilder", "NewtonSolver"]


import warp as wp
import numpy as np
import newton as _newton
from newton._src.core.reset import reset_world_selected
from newton._src.solvers.coupled.interface import CouplingInterface
from .cable_material import (
    compute_cable_dahl_parameters,
    init_cable_rest_bend_twist,
    update_cable_dahl_state,
)


@wp.kernel
def _compute_body_qdd(
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    inverse_dt: float,
    body_qdd: wp.array(dtype=wp.spatial_vector),
):
    body = wp.tid()
    body_qdd[body] = inverse_dt * (body_qd_out[body] - body_qd_in[body])


def _ensure_style3d_login() -> None:
    """Authenticate the native solver using environment variables or terminal input."""
    if is_logged_in():
        return

    account = os.getenv("STYLE3D_SIM_ACCOUNT")
    password = os.getenv("STYLE3D_SIM_PASSWORD")
    if not account or not password:
        account = input("Style3D Sim account: ").strip()
        password = getpass.getpass("Password: ")

    if not login(account, password, False):
        raise RuntimeError("Style3D Sim login failed")


def __getattr__(name: str) -> Any:
    """Load the moved builder lazily for the legacy module-level import path."""
    if name == "ModelBuilder":
        from .newton_model_builder import ModelBuilder

        globals()[name] = ModelBuilder
        return ModelBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@wp.kernel
def _refresh_affine_body_mass_kernel(
    body_mass: wp.array(dtype=float),
    body_inertia: wp.array(dtype=wp.mat33),
    body_com: wp.array(dtype=wp.vec3),
    affine_body_ids: wp.array(dtype=wp.int32),
    affine_body_indices: wp.array(dtype=wp.vec4i),
    rotation_locked: wp.array(dtype=wp.int32),
    inertia_offsets: wp.array(dtype=wp.mat33),
    mass_pair_slots: wp.array(dtype=wp.vec3i),
    mass_diagonal: wp.array(dtype=float),
    mass_non_diagonal: wp.array(dtype=float),
    affine_body_mass: wp.array(dtype=float),
    affine_body_inertia: wp.array(dtype=wp.mat33),
    affine_body_com: wp.array(dtype=wp.vec3),
):
    affine = wp.tid()
    body = affine_body_ids[affine]
    mass = body_mass[body]
    inertia = body_inertia[body]
    # Prescribed bodies and translation-only proxies retain a nonsingular dummy
    # column metric. Their authored mobility masks still prevent rotation.
    if mass <= 0.0 or rotation_locked[affine] != 0:
        inertia = wp.identity(3, dtype=float)
    else:
        inertia = 0.5 * (inertia + wp.transpose(inertia)) + inertia_offsets[affine]
    proxy_mass = wp.where(mass > 0.0, mass, 1.0)
    second_moment = 0.5 * wp.trace(inertia) * wp.identity(3, dtype=float) - inertia
    particles = affine_body_indices[affine]
    mass_diagonal[particles[0]] = proxy_mass
    for axis in range(3):
        mass_diagonal[particles[axis + 1]] = second_moment[axis, axis]
    slots = mass_pair_slots[affine]
    mass_non_diagonal[slots[0]] = second_moment[0, 1]
    mass_non_diagonal[slots[1]] = second_moment[0, 2]
    mass_non_diagonal[slots[2]] = second_moment[1, 2]
    affine_body_mass[affine] = proxy_mass
    affine_body_inertia[affine] = inertia
    affine_body_com[affine] = body_com[body]


@wp.kernel
def _predict_affine_body_inertia(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    affine_body_ids: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=wp.float32),
    affine_body_inertia: wp.array(dtype=wp.mat33),
    fixed_affine_body_flags: wp.array(dtype=wp.int32),
    inertia_mode: int,
    dt: float,
    predicted_body_q: wp.array(dtype=wp.transform),
    predicted_body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Build the selected rigid-body inertia target."""
    affine_body = wp.tid()
    body = affine_body_ids[affine_body]
    pose = body_q[body]
    if fixed_affine_body_flags[affine_body] != 0:
        predicted_body_q[body] = pose
        predicted_body_qd[body] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
        return
    rotation = wp.transform_get_rotation(pose)
    velocity = body_qd[body]
    wrench = body_f[body]
    mass = body_mass[body]
    linear_velocity = wp.spatial_top(velocity) + dt * wp.spatial_top(wrench) / mass
    angular_velocity = wp.spatial_bottom(velocity)
    body_angular_velocity = wp.quat_rotate_inv(rotation, angular_velocity)
    inertia = affine_body_inertia[affine_body]
    inverse_inertia = wp.inverse(inertia)
    initial_angular_momentum = inertia * body_angular_velocity + dt * wp.quat_rotate_inv(
        rotation, wp.spatial_bottom(wrench)
    )
    center = wp.transform_get_translation(pose) + wp.quat_rotate(
        rotation, body_com[body]
    )
    predicted_center = center + dt * linear_velocity

    if inertia_mode == VBD_SEMI_IMPLICIT:
        # VBD evaluates gyro at the original spin, before the external torque
        # kick. Passing initial_angular_momentum here would change that order.
        next_angular_momentum, relative_rotation_matrix = integrate_vbd_semi_implicit(
            inertia,
            inertia * body_angular_velocity,
            wp.quat_rotate_inv(rotation, wp.spatial_bottom(wrench)),
            dt,
        )
    else:
        next_angular_momentum, relative_rotation_matrix = integrate_rigid_body(
            inertia,
            initial_angular_momentum,
            dt,
            inertia_mode,
        )
    relative_rotation = wp.quat_from_matrix(relative_rotation_matrix)
    predicted_rotation = rotation * relative_rotation
    next_angular_velocity = wp.quat_rotate(
        predicted_rotation, inverse_inertia * next_angular_momentum
    )
    if inertia_mode == VBD_SEMI_IMPLICIT:
        next_angular_velocity = wp.quat_rotate(rotation, inverse_inertia * next_angular_momentum)
    predicted_body_q[body] = wp.transform(
        predicted_center - wp.quat_rotate(predicted_rotation, body_com[body]),
        predicted_rotation,
    )
    predicted_body_qd[body] = wp.spatial_vector(linear_velocity, next_angular_velocity)


@wp.kernel
def _predict_affine_body_velocities(
    body_q: wp.array(dtype=wp.transform),
    predicted_body_q: wp.array(dtype=wp.transform),
    affine_body_indices: wp.array(dtype=wp.vec4i),
    affine_body_ids: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    dt: float,
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Convert a Newton rigid-body prediction to direct ABD velocities."""
    affine_body = wp.tid()
    particle_indices = affine_body_indices[affine_body]
    body = affine_body_ids[affine_body]
    pose = body_q[body]
    predicted_pose = predicted_body_q[body]

    rotation = wp.transform_get_rotation(pose)
    predicted_rotation = wp.transform_get_rotation(predicted_pose)
    predicted_center = wp.transform_get_translation(predicted_pose) + wp.quat_rotate(
        predicted_rotation, body_com[body]
    )

    center_particle = particle_indices[0]
    particle_qd[center_particle] = (predicted_center - particle_q[center_particle]) / dt
    for local_axis in range(1, 4):
        particle = particle_indices[local_axis]
        current_axis = particle_q[particle]
        local_axis_vector = wp.quat_rotate_inv(rotation, current_axis)
        predicted_axis = wp.quat_rotate(predicted_rotation, local_axis_vector)
        particle_qd[particle] = (predicted_axis - current_axis) / dt


@wp.func
def _write_affine_body_state(
    affine_body: int,
    affine_body_indices: wp.array(dtype=wp.vec4i),
    affine_body_ids: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    indices = affine_body_indices[affine_body]
    body = affine_body_ids[affine_body]
    pose = body_q[body]
    rotation = wp.transform_get_rotation(pose)
    world_com = wp.transform_get_translation(pose) + wp.quat_rotate(
        rotation, body_com[affine_body]
    )
    x_axis = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
    y_axis = wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0))
    z_axis = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    if particle_q:
        particle_q[indices[0]] = world_com
        particle_q[indices[1]] = x_axis
        particle_q[indices[2]] = y_axis
        particle_q[indices[3]] = z_axis
    if particle_qd:
        linear_velocity = wp.spatial_top(body_qd[body])
        angular_velocity = wp.spatial_bottom(body_qd[body])
        particle_qd[indices[0]] = linear_velocity
        particle_qd[indices[1]] = wp.cross(angular_velocity, x_axis)
        particle_qd[indices[2]] = wp.cross(angular_velocity, y_axis)
        particle_qd[indices[3]] = wp.cross(angular_velocity, z_axis)


@wp.kernel
def _sync_affine_body_state(
    reference_particle_q: wp.array(dtype=wp.vec3),
    affine_body_indices: wp.array(dtype=wp.vec4i),
    affine_body_ids: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_mask: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    """Map authored Newton body poses and twists to direct ABD coordinates."""
    if body_mask and body_mask[affine_body_ids[wp.tid()]] == 0:
        return
    _write_affine_body_state(
        wp.tid(), affine_body_indices, affine_body_ids, body_com, body_q, body_qd, particle_q, particle_qd
    )


@wp.kernel
def _sync_selected_affine_body_state(
    affine_body_indices: wp.array(dtype=wp.vec4i),
    affine_body_ids: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_world: wp.array(dtype=wp.int32),
    world_mask: wp.array(dtype=wp.bool),
    world_count: int,
    reset_all: bool,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    affine_body = wp.tid()
    if not reset_all:
        if not reset_world_selected(body_world[affine_body_ids[affine_body]], world_mask, world_count):
            return
    _write_affine_body_state(
        affine_body, affine_body_indices, affine_body_ids, body_com, body_q, body_qd, particle_q, particle_qd
    )


@wp.func
def _reset_selected_world(world: int, world_mask: wp.array(dtype=wp.bool), world_count: int, reset_all: bool):
    if reset_all:
        return True
    return reset_world_selected(world, world_mask, world_count)


@wp.kernel
def _reset_newton_state(
    world_mask: wp.array(dtype=wp.bool),
    reset_all: bool,
    world_count: int,
    body_world: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=wp.int32),
    model_body_q: wp.array(dtype=wp.transform),
    model_body_qd: wp.array(dtype=wp.spatial_vector),
    model_particle_q: wp.array(dtype=wp.vec3),
    model_particle_qd: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    if (body_q or body_qd) and index < body_world.shape[0]:
        if _reset_selected_world(body_world[index], world_mask, world_count, reset_all):
            if body_q:
                body_q[index] = model_body_q[index]
            if body_qd:
                body_qd[index] = model_body_qd[index]
    if (particle_q or particle_qd) and index < particle_world.shape[0]:
        if _reset_selected_world(particle_world[index], world_mask, world_count, reset_all):
            if particle_q:
                particle_q[index] = model_particle_q[index]
            if particle_qd:
                particle_qd[index] = model_particle_qd[index]


@wp.kernel
def _reset_cable_history(
    world_mask: wp.array(dtype=wp.bool),
    reset_all: bool,
    world_count: int,
    cable_joints: wp.array(dtype=wp.int32),
    joint_world: wp.array(dtype=wp.int32),
    rebaseline_mask: wp.array(dtype=wp.bool),
    sigma_prev: wp.array(dtype=wp.vec3),
    kappa_prev: wp.array(dtype=wp.vec3),
    dkappa_prev: wp.array(dtype=wp.vec3),
    sigma_start: wp.array(dtype=wp.vec3),
    friction_hessian: wp.array(dtype=wp.vec3),
    penalty_reset_mask: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    if index < world_count + 1:
        world = index
        if index == world_count:
            world = -1
        if _reset_selected_world(world, world_mask, world_count, reset_all):
            rebaseline_mask[index] = True
    if index < joint_world.shape[0]:
        if _reset_selected_world(joint_world[index], world_mask, world_count, reset_all):
            # Newton rods include joint rows that do not carry cable state; clear the full selected-world slice.
            sigma_prev[index] = wp.vec3(0.0)
            kappa_prev[index] = wp.vec3(0.0)
            dkappa_prev[index] = wp.vec3(0.0)
            sigma_start[index] = wp.vec3(0.0)
            friction_hessian[index] = wp.vec3(0.0)
    if index < cable_joints.shape[0]:
        joint = cable_joints[index]
        selected = _reset_selected_world(joint_world[joint], world_mask, world_count, reset_all)
        penalty_reset_mask[index] = wp.int32(selected)


@wp.kernel
def _gather_cable_parameters(
    cable_joints: wp.array(dtype=wp.int32),
    joint_enabled: wp.array(dtype=wp.bool),
    joint_kb_rest: wp.array(dtype=wp.vec3),
    joint_twist_rest: wp.array(dtype=wp.float32),
    joint_sigma_start: wp.array(dtype=wp.vec3),
    joint_C_fric: wp.array(dtype=wp.vec3),
    cable_enabled: wp.array(dtype=wp.int32),
    cable_rest: wp.array(dtype=wp.vec3),
    cable_bias: wp.array(dtype=wp.vec3),
    cable_hessian: wp.array(dtype=wp.vec3),
):
    cable = wp.tid()
    joint = cable_joints[cable]
    kb = joint_kb_rest[joint]
    cable_enabled[cable] = wp.int32(joint_enabled[joint])
    cable_rest[cable] = wp.vec3(kb[0], kb[1], joint_twist_rest[joint])
    cable_bias[cable] = joint_sigma_start[joint]
    cable_hessian[cable] = joint_C_fric[joint]


@wp.kernel
def _sync_fixed_affine_body_state(
    affine_body_indices: wp.array(dtype=wp.vec4i),
    affine_body_ids: wp.array(dtype=wp.int32),
    fixed_affine_body_flags: wp.array(dtype=wp.int32),
    rotation_locked_affine_body_flags: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    dt: float,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    affine = wp.tid()
    fixed = fixed_affine_body_flags[affine] != 0
    if not fixed and rotation_locked_affine_body_flags[affine] == 0:
        return
    body = affine_body_ids[affine]
    indices = affine_body_indices[affine]
    pose = body_q[body]
    rotation = wp.transform_get_rotation(pose)
    origin = wp.transform_point(pose, body_com[affine])
    axis_x = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
    axis_y = wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0))
    axis_z = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    for corner in range(4):
        if corner == 0 and not fixed:
            continue
        particle = indices[corner]
        current = origin
        if corner == 1:
            current = axis_x
        elif corner == 2:
            current = axis_y
        elif corner == 3:
            current = axis_z
        # Contact friction needs the prescribed surface displacement even though
        # inactive proxies must retain their newly authored positions in the solve.
        particle_qd[particle] = (current - particle_q[particle]) / dt
        particle_q[particle] = current


@wp.kernel
def _mask_joint_drive_reset(
    body_mask: wp.array(dtype=wp.int32),
    affine_body_ids: wp.array(dtype=wp.int32),
    drive_bodies: wp.array(dtype=wp.vec2i),
    angle_initialized: wp.array(dtype=wp.int32),
    drive_mask: wp.array(dtype=wp.int32),
):
    drive = wp.tid()
    bodies = drive_bodies[drive]
    selected = body_mask[affine_body_ids[bodies[1]]] != 0
    if bodies[0] >= 0:
        selected = selected or body_mask[affine_body_ids[bodies[0]]] != 0
    drive_mask[drive] = int(selected)
    if selected:
        angle_initialized[drive] = 0


@wp.kernel
def _update_affine_body_poses(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    affine_body_indices: wp.array(dtype=wp.vec4i),
    affine_body_ids: wp.array(dtype=wp.int32),
    fixed_affine_body_flags: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    affine_body_inertia: wp.array(dtype=wp.mat33),
    inertia_mode: int,
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Reconstruct Newton body state and canonicalize NONE proxy velocities."""
    affine_body = wp.tid()
    if fixed_affine_body_flags[affine_body] != 0:
        return
    particle_indices = affine_body_indices[affine_body]
    body = affine_body_ids[affine_body]
    origin = particle_q[particle_indices[0]]

    affine_matrix = wp.matrix_from_cols(
        particle_q[particle_indices[1]],
        particle_q[particle_indices[2]],
        particle_q[particle_indices[3]],
    )
    # Extract the closest rotation in the authored ABD mass metric. The affine
    # kinetic energy weights frame columns by the second moment D, so the
    # weighted Procrustes problem maximizes tr(R^T A D), not tr(R^T A).
    inertia = affine_body_inertia[affine_body]
    second_moment = 0.5 * wp.trace(inertia) * wp.identity(3, dtype=float) - inertia
    # Keep the mass-weighted rotation extraction in double precision: a float32
    # SVD can rotate even an identity frame, and endpoint velocity amplifies the
    # resulting angular error by 1/dt. Do not clamp small physical rotations.
    inertia64 = wp.mat33d(inertia)
    second_moment64 = wp.float64(0.5) * wp.trace(inertia64) * wp.identity(3, dtype=wp.float64) - inertia64
    weighted_affine_matrix = wp.mat33d(affine_matrix) * second_moment64
    # Warp's SVD uses absolute QR tolerances. Normalize before decomposition so
    # small physical inertias cannot turn a stationary frame into a false spin.
    # A positive uniform factor preserves the mass-weighted Procrustes solution.
    matrix_scale = wp.float64(0.0)
    for row in range(3):
        for column in range(3):
            matrix_scale = wp.max(matrix_scale, wp.abs(weighted_affine_matrix[row, column]))
    if matrix_scale <= wp.float64(0.0):
        return
    left_basis, singular_values, right_basis = wp.svd3(weighted_affine_matrix / matrix_scale)
    if wp.abs(singular_values[2]) <= wp.float64(1.0e-8) * wp.abs(singular_values[0]):
        return

    polar_rotation = left_basis * wp.transpose(right_basis)
    if wp.determinant(polar_rotation) < wp.float64(0.0):
        left_basis = wp.matrix_from_cols(
            wp.vec3d(left_basis[0, 0], left_basis[1, 0], left_basis[2, 0]),
            wp.vec3d(left_basis[0, 1], left_basis[1, 1], left_basis[2, 1]),
            -wp.vec3d(left_basis[0, 2], left_basis[1, 2], left_basis[2, 2]),
        )
        polar_rotation = left_basis * wp.transpose(right_basis)

    rotation = wp.quat(wp.quat_from_matrix(polar_rotation))
    polar_rotation32 = wp.mat33(polar_rotation)
    previous_rotation = wp.transform_get_rotation(body_q[body])
    quaternion_dot = (
        rotation[0] * previous_rotation[0]
        + rotation[1] * previous_rotation[1]
        + rotation[2] * previous_rotation[2]
        + rotation[3] * previous_rotation[3]
    )
    # q and -q encode the same rotation, but Newton's inverse kinematics uses
    # the quaternion branch to recover signed revolute coordinates. Preserve
    # the previous hemisphere so a small negative angle cannot appear as 2*pi.
    if quaternion_dot < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])
    center_velocity = particle_qd[particle_indices[0]]
    angular_velocity = wp.vec3(0.0)
    if inertia_mode != NONE:
        # Recover Q = R_n^T R_{n+1} and apply the endpoint closure paired with
        # the selected rigid-body inertia integrator.
        relative_rotation = wp.quat_inverse(previous_rotation) * rotation
        if relative_rotation[3] < 0.0:
            relative_rotation = wp.quat(
                -relative_rotation[0],
                -relative_rotation[1],
                -relative_rotation[2],
                -relative_rotation[3],
            )
        next_body_angular_velocity = recover_rigid_body_end_velocity(
            inertia,
            wp.quat_to_matrix(relative_rotation),
            dt,
            inertia_mode,
        )
        angular_velocity = wp.quat_rotate(rotation, next_body_angular_velocity)
    else:
        # The legacy free-particle inertia path has no rigid-body rotation map
        # to invert. Project its converged affine velocity gradient onto the
        # closest rigid angular velocity in the authored mass metric instead.
        velocity_gradient = wp.matrix_from_cols(
            particle_qd[particle_indices[1]],
            particle_qd[particle_indices[2]],
            particle_qd[particle_indices[3]],
        )
        coupling = velocity_gradient * second_moment * wp.transpose(polar_rotation32)
        angular_momentum = wp.vec3(
            coupling[2, 1] - coupling[1, 2],
            coupling[0, 2] - coupling[2, 0],
            coupling[1, 0] - coupling[0, 1],
        )
        world_inertia = polar_rotation32 * inertia * wp.transpose(polar_rotation32)
        angular_velocity = wp.inverse(world_inertia) * angular_momentum

        # body_qd stores only a rigid twist, so make the externally visible
        # proxy velocities the exact rigid field represented by that twist.
        # This makes NONE's velocity projection idempotent: recovering body_qd
        # and expanding it again produces the same proxy velocities.
        x_axis = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
        y_axis = wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0))
        z_axis = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
        particle_qd[particle_indices[0]] = center_velocity
        particle_qd[particle_indices[1]] = wp.cross(angular_velocity, x_axis)
        particle_qd[particle_indices[2]] = wp.cross(angular_velocity, y_axis)
        particle_qd[particle_indices[3]] = wp.cross(angular_velocity, z_axis)
    body_q[body] = wp.transform(
        origin - wp.quat_rotate(rotation, body_com[body]), rotation
    )
    body_qd[body] = wp.spatial_vector(center_velocity, angular_velocity)


@wp.kernel
def _scatter_affine_body_forces(
    body_f: wp.array(dtype=wp.spatial_vector),
    affine_body_indices: wp.array(dtype=wp.vec4i),
    affine_body_ids: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
):
    """Map a Newton body wrench to direct ABD generalized forces."""
    affine_body = wp.tid()
    particle_indices = affine_body_indices[affine_body]
    body = affine_body_ids[affine_body]
    wrench = body_f[body]
    force = wp.spatial_top(wrench)
    torque = wp.spatial_bottom(wrench)
    x_axis = particle_q[particle_indices[1]]
    y_axis = particle_q[particle_indices[2]]
    z_axis = particle_q[particle_indices[3]]
    force_x = 0.5 * wp.cross(torque, x_axis)
    force_y = 0.5 * wp.cross(torque, y_axis)
    force_z = 0.5 * wp.cross(torque, z_axis)
    wp.atomic_add(particle_f, particle_indices[0], force)
    wp.atomic_add(particle_f, particle_indices[1], force_x)
    wp.atomic_add(particle_f, particle_indices[2], force_y)
    wp.atomic_add(particle_f, particle_indices[3], force_z)


@wp.kernel
def _clear_fixed_affine_body_velocities(
    particle_qd: wp.array(dtype=wp.vec3),
    affine_body_indices: wp.array(dtype=wp.vec4i),
    fixed_affine_bodies: wp.array(dtype=wp.int32),
):
    """Keep inactive affine proxy velocities consistent with their fixed positions."""
    fixed_body = wp.tid()
    particle_indices = affine_body_indices[fixed_affine_bodies[fixed_body]]
    particle_qd[particle_indices[0]] = wp.vec3(0.0)
    particle_qd[particle_indices[1]] = wp.vec3(0.0)
    particle_qd[particle_indices[2]] = wp.vec3(0.0)
    particle_qd[particle_indices[3]] = wp.vec3(0.0)


class NewtonSolver(_newton.solvers.SolverBase, CouplingInterface):
    """Expose the native Style3D Sim solver through Newton's solver interface.

    The adapter passes Newton-owned Warp CUDA arrays to Style3D Sim as non-owning
    device-memory spans. Newton continues to own every model and state array.

    The native solver supports particles, tetrahedral soft bodies, Style3D
    cloth, dynamic affine bodies, and rigid and soft contacts from Newton's
    collision pipeline.
    Newton ball, revolute, prismatic, fixed, free, and equivalent D6 joints are
    assembled from affine-body point-point and point-line constraints. Cable
    joints retain their stretch, shear, discrete bend/twist, and Dahl materials
    in both the Newton and projective-dynamics frameworks.
    Joint controls and Newton collision-pipeline contacts are passed to the
    native backend. Native work uses Newton's active CUDA stream and can
    therefore participate in graph capture.
    """

    SUPPORTED_OBSERVABLES = frozenset(
        {
            _newton.solvers.SolverBase.ObservableKind.BODY_QDD,
            _newton.solvers.SolverBase.ObservableKind.CONTACT_F,
        }
    )

    def __init__(
        self,
        model: Any,
        *,
        joint_damping: float = 100.0,
        joint_attachment_distance: float = 0.1,
        joint_line_half_length: float = 10.0,
        joint_drive_armature: float = 0.0,
    ) -> None:
        """Create a Style3D Sim-backed solver for a Newton model.

        Args:
            model: Particle, tetrahedral soft-body, or Style3D cloth
                ``newton.Model`` on a CUDA device.
            joint_damping: Normal damping used for generated point-line constraints.
            joint_attachment_distance: Separation between attachment points that
                constrain joint orientation.
            joint_line_half_length: Half-length of generated slider guide segments.
            joint_drive_armature: Isotropic rotational inertia added to bodies
                connected to active joint drives. This optional legacy
                regularization is separate from ``model.joint_armature``,
                which adds inertia to each supported relative joint coordinate.

        Raises:
            ValueError: If the model is not allocated on CUDA or required
                particle or tetrahedron arrays are invalid.
            RuntimeError: If Style3D Sim authentication fails.
            NotImplementedError: If generic Newton surface elasticity is
                enabled without Style3D attributes.
        """
        for name, value, allow_zero in (
            ("joint_damping", joint_damping, True),
            ("joint_attachment_distance", joint_attachment_distance, False),
            ("joint_line_half_length", joint_line_half_length, False),
            ("joint_drive_armature", joint_drive_armature, True),
        ):
            if not np.isfinite(value) or (
                float(value) < 0.0 if allow_zero else float(value) <= 0.0
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}.")

        super().__init__(model)
        _ensure_style3d_login()
        self._wp = import_module("warp")
        self._joint_drive_effort: Any | None = None
        # Feedback records the same pre-integration actuator evaluation as MuJoCo.
        self._joint_drive_feedback_state: Any | None = None
        self._joint_drive_feedback_previous_state: Any | None = None
        self._joint_drive_feedback_dt: float | None = None
        self._joint_drive_feedback_previous_angles: Any | None = None
        self._joint_drive_feedback_angle_initialized: Any | None = None
        self._joint_drive_reset_mask: Any | None = None
        self._joint_drive_feedback_scratch: Any | None = None
        self._joint_drive_start_velocities: Any | None = None
        active_joint_drive_body_ids = self._find_active_joint_drive_body_ids()
        self._joint_drive_armature = float(joint_drive_armature)
        self._pre_regularized_body_ids: set[int] = set()
        self._ensure_affine_body_proxies(
            armature=float(joint_drive_armature),
            armature_body_ids=active_joint_drive_body_ids,
        )
        self._body_prediction_state = SimpleNamespace(
            body_q=wp.empty_like(model.body_q),
            body_qd=wp.empty_like(model.body_qd),
        )
        self._body_qd_previous = wp.empty_like(model.body_qd)
        self._particle_mass_diagonal: Any | None = getattr(
            model, "synreal_particle_mass_diagonal", None
        )
        self._particle_mass_non_diagonal_indices: Any | None = getattr(
            model, "synreal_particle_mass_non_diagonal_indices", None
        )
        self._particle_mass_non_diagonal_values: Any | None = getattr(
            model, "synreal_particle_mass_non_diagonal_values", None
        )
        self._tetrahedron_indices: Any | None = getattr(model, "tet_indices", None)
        self._tetrahedron_inverse_dms: Any | None = getattr(model, "tet_poses", None)
        self._tetrahedron_activations: Any | None = getattr(
            model, "tet_activations", None
        )
        self._tetrahedron_materials: Any | None = getattr(model, "tet_materials", None)
        self._style3d = getattr(model, "style3d", None)
        self._native_triangle_count = (
            int(model.tri_count) if self._style3d is not None else 0
        )
        self._native_edge_count = (
            int(model.edge_count) if self._style3d is not None else 0
        )
        self._collision_triangle_count = int(model.tri_count)
        self._collision_edge_indices = self._build_collision_edge_indices()
        self._collision_edge_count = 0 if self._collision_edge_indices is None else len(self._collision_edge_indices)
        self._triangle_drag_coefficients = self._extract_triangle_drag_coefficients()
        self._soft_particle_only_shapes = self._extract_particle_only_shapes()
        self._affine_body_ids: Any | None = getattr(
            model, "synreal_affine_body_ids", None
        )
        self._affine_body_indices: Any | None = getattr(
            model, "synreal_affine_body_indices", None
        )
        self._affine_body_masses: Any | None = getattr(
            model, "synreal_affine_body_masses", None
        )
        self._affine_body_inertia: Any | None = getattr(
            model, "synreal_affine_body_inertia", None
        )
        self._affine_body_com: Any | None = getattr(
            model, "synreal_affine_body_com", None
        )
        self._body_spring_indices: Any | None = getattr(
            model, "synreal_body_spring_indices", None
        )
        self._body_spring_points0: Any | None = getattr(
            model, "synreal_body_spring_points0", None
        )
        self._body_spring_points1: Any | None = getattr(
            model, "synreal_body_spring_points1", None
        )
        self._body_spring_stiffnesses: Any | None = getattr(
            model, "synreal_body_spring_stiffnesses", None
        )
        self._body_spring_rest_lengths: Any | None = getattr(
            model, "synreal_body_spring_rest_lengths", None
        )
        self._body_distance_constraint_indices: Any | None = getattr(
            model, "synreal_body_distance_constraint_indices", None
        )
        self._body_distance_constraint_points0: Any | None = getattr(
            model, "synreal_body_distance_constraint_points0", None
        )
        self._body_distance_constraint_points1: Any | None = getattr(
            model, "synreal_body_distance_constraint_points1", None
        )
        self._body_distance_constraint_weights0: Any | None = getattr(
            model, "synreal_body_distance_constraint_weights0", None
        )
        self._body_distance_constraint_weights1: Any | None = getattr(
            model, "synreal_body_distance_constraint_weights1", None
        )
        self._body_distance_constraint_rest_lengths: Any | None = getattr(
            model, "synreal_body_distance_constraint_rest_lengths", None
        )
        self._body_distance_constraint_mass_scales: Any | None = None
        self._body_point_line_indices: Any | None = getattr(
            model, "synreal_body_point_line_indices", None
        )
        self._body_point_line_points: Any | None = getattr(
            model, "synreal_body_point_line_points", None
        )
        self._body_point_line_endpoints0: Any | None = getattr(
            model, "synreal_body_point_line_endpoints0", None
        )
        self._body_point_line_endpoints1: Any | None = getattr(
            model, "synreal_body_point_line_endpoints1", None
        )
        self._body_point_line_stiffnesses: Any | None = getattr(
            model, "synreal_body_point_line_stiffnesses", None
        )
        self._body_point_line_dampings: Any | None = getattr(
            model, "synreal_body_point_line_dampings", None
        )
        self._joint_body_spring_count = 0
        self._joint_body_point_line_count = 0
        self._joint_constraint_mass_scale = 1.0
        self._joint_drive_types: Any | None = None
        self._joint_drive_body_indices: Any | None = None
        self._joint_drive_parent_origins: Any | None = None
        self._joint_drive_child_origins: Any | None = None
        self._joint_drive_parent_axis_points: Any | None = None
        self._joint_drive_parent_reference_points: Any | None = None
        self._joint_drive_child_reference_points: Any | None = None
        self._joint_drive_dof_indices: Any | None = None
        self._joint_drive_target_indices: Any | None = None
        self._fixed_affine_body_ids = self._find_fixed_affine_bodies()
        fixed_affine_body_set = set(self._fixed_affine_body_ids)
        affine_body_count = (
            0 if self._affine_body_ids is None else len(self._affine_body_ids)
        )
        self._fixed_affine_body_flags = wp.array(
            [
                int(affine_body in fixed_affine_body_set)
                for affine_body in range(affine_body_count)
            ],
            dtype=wp.int32,
            device=model.device,
        )
        self._rotation_locked_affine_body_ids: list[int] = []
        self._rotation_locked_affine_body_flags = wp.zeros(affine_body_count, dtype=wp.int32, device=model.device)
        self._fixed_affine_bodies = (
            None
            if not self._fixed_affine_body_ids
            else wp.array(
                self._fixed_affine_body_ids,
                dtype=wp.int32,
                device=model.device,
            )
        )
        self._particle_flags = self._build_particle_flags()
        self._collision_particle_indices = self._build_collision_particle_indices()
        self._collision_particle_count = len(self._collision_particle_indices)
        self._body_distance_constraint_mass_scales = (
            self._normalize_body_distance_constraint_mass_scales()
        )
        self._assemble_joint_constraints(
            damping=float(joint_damping),
            attachment_distance=float(joint_attachment_distance),
            line_half_length=float(joint_line_half_length),
            drive_armature=float(joint_drive_armature),
        )
        self._assemble_cable_constraints()
        self._prepare_affine_inertial_refresh()
        self._validate_model()
        self._configure_shared_world_layout()

        self._particle_indices: Any | None = None
        self._synreal_model = Model()
        # Host array reads may restore a different current CUDA context. Native
        # Thrust buffers must be allocated in the Newton model's device context.
        with wp.ScopedDevice(model.device):
            self._native_solver = Solver()
        self._rigid_contact_capacity = 0
        self._initialize_native_solver()
        # Keep the Python angle branch in lockstep with native joint-drive
        # state even while public effort telemetry is disabled.  This makes
        # late enable/re-enable observe the same multi-turn coordinate and
        # ensures graph capture never allocates branch buffers on demand.
        self._ensure_joint_drive_feedback_angle_state()

    def _configure_shared_world_layout(self) -> None:
        """Use one static model template for homogeneous replicated worlds."""
        world_count = int(self.model.world_count)
        self._shared_world_count = 1
        self._shared_counts: dict[str, int] = {}
        self._shared_shape_arrays: dict[str, Any] = {}
        if self._cable_count:
            return
        # Material selections use global Newton indices; preserve them and let the
        # native hash queries isolate worlds using particle_world.
        if any(
            getattr(self.model, f"synreal_{material}_particle_indices", None) is not None
            for material in ("fluid", "granular")
        ):
            return
        if world_count <= 1:
            return

        particle_world = np.asarray(self.model.particle_world.numpy(), dtype=np.int64)
        particle_count = int(self.model.particle_count)
        body_count = 0 if self._affine_body_ids is None else len(self._affine_body_ids)
        if (
            particle_count == 0
            or particle_count % world_count
            or body_count % world_count
        ):
            return
        particles_per_world = particle_count // world_count
        bodies_per_world = body_count // world_count
        expected_particle_world = np.repeat(np.arange(world_count), particles_per_world)
        if not np.array_equal(particle_world, expected_particle_world):
            return
        body_world = np.asarray(self.model.body_world.numpy(), dtype=np.int64)
        if not np.array_equal(
            body_world, np.repeat(np.arange(world_count), bodies_per_world)
        ):
            return

        def has_repeated_blocks(array: Any, block_size: int) -> bool:
            values = np.asarray(array.numpy())
            if len(values) != block_size * world_count:
                return False
            blocks = values.reshape((world_count, block_size, *values.shape[1:]))
            return all(np.array_equal(block, blocks[0]) for block in blocks[1:])

        def has_repeated_index_blocks(
            array: Any, block_size: int, index_stride: int
        ) -> bool:
            values = np.asarray(array.numpy(), dtype=np.int64)
            state_count = block_size * world_count
            if state_count == 0 or values.size % state_count:
                return False
            components = values.size // state_count
            blocks = values.reshape((world_count, block_size, components))
            first = blocks[0]
            if np.any(first < -1) or np.any(first >= index_stride):
                return False
            for world, block in enumerate(blocks[1:], start=1):
                normalized = np.where(
                    block >= 0,
                    block - world * index_stride,
                    block,
                )
                if not np.array_equal(normalized, first):
                    return False
            return True

        mass_diagonal = (
            self._particle_mass_diagonal
            if self._particle_mass_diagonal is not None
            else self.model.particle_mass
        )
        if not all(
            has_repeated_blocks(array, particles_per_world)
            for array in (
                mass_diagonal,
                self.model.particle_radius,
                self._particle_flags,
            )
        ):
            return
        if body_count and not all(
            has_repeated_blocks(array, bodies_per_world)
            for array in (
                self._affine_body_masses,
                self._affine_body_inertia,
                self._affine_body_com,
            )
        ):
            return

        def per_world(total: int) -> int | None:
            return total // world_count if total % world_count == 0 else None

        count_sources = {
            "particle_count": particle_count,
            "particle_mass_non_diagonal_count": (
                0
                if self._particle_mass_non_diagonal_indices is None
                else len(self._particle_mass_non_diagonal_indices)
            ),
            "spring_count": int(self.model.spring_count),
            "tetrahedron_count": int(self.model.tet_count),
            "triangle_count": self._native_triangle_count,
            "edge_count": self._native_edge_count,
            "collision_triangle_count": self._collision_triangle_count,
            "collision_edge_count": self._collision_edge_count,
            "collision_particle_count": self._collision_particle_count,
            "body_count": body_count,
            "body_spring_count": (
                0
                if self._body_spring_indices is None
                else len(self._body_spring_indices)
            ),
            "joint_body_spring_count": self._joint_body_spring_count,
            "body_distance_constraint_count": (
                0
                if self._body_distance_constraint_indices is None
                else len(self._body_distance_constraint_indices)
            ),
            "body_point_line_count": (
                0
                if self._body_point_line_indices is None
                else len(self._body_point_line_indices)
            ),
            "joint_body_point_line_count": self._joint_body_point_line_count,
            "joint_drive_count": (
                0 if self._joint_drive_types is None else len(self._joint_drive_types)
            ),
            "joint_dof_count": int(self.model.joint_dof_count),
            "joint_target_count": len(self.model.joint_target_q),
        }
        local_counts = {name: per_world(total) for name, total in count_sources.items()}
        if any(count is None for count in local_counts.values()):
            return

        # Generated joint constraints form a suffix after authored constraints.
        # A mixed prefix/suffix is not one contiguous per-world template, while
        # either family by itself can be shared after validating every block.
        for total_name, joint_name in (
            ("body_spring_count", "joint_body_spring_count"),
            ("body_point_line_count", "joint_body_point_line_count"),
        ):
            total = count_sources[total_name]
            joint = count_sources[joint_name]
            if joint not in (0, total):
                return

        deformable_specs = (
            (
                "spring_count",
                self.model.spring_indices,
                (
                    self.model.spring_stiffness,
                    self.model.spring_rest_length,
                ),
            ),
            (
                "tetrahedron_count",
                self._tetrahedron_indices,
                (
                    self._tetrahedron_inverse_dms,
                    self._tetrahedron_materials,
                ),
            ),
            (
                "triangle_count",
                self.model.tri_indices,
                (
                    self.model.tri_areas,
                    self.model.tri_poses,
                    None if self._style3d is None else self._style3d.tri_aniso_ke,
                ),
            ),
            (
                "edge_count",
                self.model.edge_indices,
                (
                    self.model.edge_bending_properties,
                    None if self._style3d is None else self._style3d.edge_rest_area,
                    None if self._style3d is None else self._style3d.edge_bending_cot,
                ),
            ),
            (
                "collision_triangle_count",
                self.model.tri_indices,
                (() if self._triangle_drag_coefficients is None else (self._triangle_drag_coefficients,)),
            ),
            (
                "collision_edge_count",
                self._collision_edge_indices,
                (),
            ),
            (
                "collision_particle_count",
                self._collision_particle_indices,
                (),
            ),
        )
        for count_name, indices, attributes in deformable_specs:
            count = local_counts[count_name]
            if not count:
                continue
            if indices is None or any(attribute is None for attribute in attributes):
                return
            if not has_repeated_index_blocks(indices, count, particles_per_world):
                return
            if not all(
                has_repeated_blocks(attribute, count) for attribute in attributes
            ):
                return

        mass_non_diagonal_count = local_counts["particle_mass_non_diagonal_count"]
        if mass_non_diagonal_count and (
            not has_repeated_index_blocks(
                self._particle_mass_non_diagonal_indices,
                mass_non_diagonal_count,
                particles_per_world,
            )
            or not has_repeated_blocks(
                self._particle_mass_non_diagonal_values,
                mass_non_diagonal_count,
            )
        ):
            return

        if body_count and not has_repeated_index_blocks(
            self._affine_body_indices,
            bodies_per_world,
            particles_per_world,
        ):
            return

        body_springs_per_world = local_counts["body_spring_count"]
        if body_springs_per_world:
            if not has_repeated_index_blocks(
                self._body_spring_indices,
                body_springs_per_world,
                bodies_per_world,
            ) or not all(
                has_repeated_blocks(array, body_springs_per_world)
                for array in (
                    self._body_spring_points0,
                    self._body_spring_points1,
                    self._body_spring_stiffnesses,
                    self._body_spring_rest_lengths,
                )
            ):
                return

        body_point_lines_per_world = local_counts["body_point_line_count"]
        if body_point_lines_per_world:
            if not has_repeated_index_blocks(
                self._body_point_line_indices,
                body_point_lines_per_world,
                bodies_per_world,
            ) or not all(
                has_repeated_blocks(array, body_point_lines_per_world)
                for array in (
                    self._body_point_line_points,
                    self._body_point_line_endpoints0,
                    self._body_point_line_endpoints1,
                    self._body_point_line_stiffnesses,
                    self._body_point_line_dampings,
                )
            ):
                return

        joint_drives_per_world = local_counts["joint_drive_count"]
        if joint_drives_per_world:
            joint_dofs_per_world = local_counts["joint_dof_count"]
            joint_targets_per_world = local_counts["joint_target_count"]
            if (
                not has_repeated_blocks(
                    self._joint_drive_types, joint_drives_per_world
                )
                or not has_repeated_index_blocks(
                    self._joint_drive_body_indices,
                    joint_drives_per_world,
                    bodies_per_world,
                )
                or not has_repeated_index_blocks(
                    self._joint_drive_dof_indices,
                    joint_drives_per_world,
                    joint_dofs_per_world,
                )
                or not has_repeated_index_blocks(
                    self._joint_drive_target_indices,
                    joint_drives_per_world,
                    joint_targets_per_world,
                )
                or not all(
                    has_repeated_blocks(array, joint_drives_per_world)
                    for array in (
                        self._joint_drive_parent_origins,
                        self._joint_drive_child_origins,
                        self._joint_drive_parent_axis_points,
                        self._joint_drive_parent_reference_points,
                        self._joint_drive_child_reference_points,
                    )
                )
                or not all(
                    has_repeated_blocks(array, joint_dofs_per_world)
                    for array in (
                        self.model.joint_target_ke,
                        self.model.joint_target_kd,
                        self.model.joint_target_mode,
                        self.model.joint_damping,
                        self.model.joint_armature,
                        self.model.joint_effort_limit,
                        self.model.joint_limit_lower,
                        self.model.joint_limit_upper,
                        self.model.joint_limit_ke,
                        self.model.joint_limit_kd,
                    )
                )
            ):
                return

        distance_constraints_per_world = local_counts["body_distance_constraint_count"]
        if distance_constraints_per_world:
            constraint_indices = np.asarray(
                self._body_distance_constraint_indices.numpy(), dtype=np.int64
            ).reshape((world_count, distance_constraints_per_world, 2))
            first_indices = constraint_indices[0]
            for world, block in enumerate(constraint_indices[1:], start=1):
                normalized = np.where(
                    block >= 0,
                    block - world * bodies_per_world,
                    block,
                )
                if not np.array_equal(normalized, first_indices):
                    return
            if not all(
                has_repeated_blocks(array, distance_constraints_per_world)
                for array in (
                    self._body_distance_constraint_points0,
                    self._body_distance_constraint_points1,
                    self._body_distance_constraint_weights0,
                    self._body_distance_constraint_weights1,
                    self._body_distance_constraint_rest_lengths,
                    self._body_distance_constraint_mass_scales,
                )
            ):
                return

        # Newton appends global shapes (the H1 ground plane is one) after all
        # replicated shapes. Keep that suffix once beside the first world's shapes.
        shape_bodies = np.asarray(self.model.shape_body.numpy(), dtype=np.int32)
        global_shape_count = 0
        for body in shape_bodies[::-1]:
            if int(body) != -1:
                break
            global_shape_count += 1
        replicated_shape_total = len(shape_bodies) - global_shape_count
        if replicated_shape_total % world_count:
            return
        shapes_per_world = replicated_shape_total // world_count
        if np.any(shape_bodies[:replicated_shape_total] < 0):
            return
        newton_bodies_per_world = int(self.model.body_count) // world_count
        first_shape_bodies = shape_bodies[:shapes_per_world]
        for world in range(world_count):
            start = world * shapes_per_world
            if not np.array_equal(
                shape_bodies[start : start + shapes_per_world],
                first_shape_bodies + world * newton_bodies_per_world,
            ):
                return
        for name in (
            "shape_type",
            "shape_transform",
            "shape_scale",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_mu",
            "shape_material_mu_rolling",
            "shape_material_mu_torsional",
            "shape_margin",
        ):
            values = np.asarray(getattr(self.model, name).numpy())[
                :replicated_shape_total
            ]
            blocks = values.reshape((world_count, shapes_per_world, *values.shape[1:]))
            if any(not np.array_equal(block, blocks[0]) for block in blocks[1:]):
                return

        self._shared_world_count = world_count
        self._shared_counts = {name: int(count) for name, count in local_counts.items()}
        self._shared_counts["shape_replicated_count"] = shapes_per_world
        self._shared_counts["shape_count"] = shapes_per_world + global_shape_count

        for native_name, model_name, dtype in (
            ("shape_bodies", "shape_body", wp.int32),
            ("shape_types", "shape_type", wp.int32),
            ("shape_transforms", "shape_transform", wp.transform),
            ("shape_scales", "shape_scale", wp.vec3),
            ("shape_contact_stiffnesses", "shape_material_ke", wp.float32),
            ("shape_contact_dampings", "shape_material_kd", wp.float32),
            ("shape_contact_friction_gains", "shape_material_kf", wp.float32),
            ("shape_frictions", "shape_material_mu", wp.float32),
            ("shape_rolling_frictions", "shape_material_mu_rolling", wp.float32),
            ("shape_torsional_frictions", "shape_material_mu_torsional", wp.float32),
            ("shape_margins", "shape_margin", wp.float32),
        ):
            source = np.asarray(getattr(self.model, model_name).numpy())
            compact = (
                np.concatenate(
                    (source[:shapes_per_world], source[-global_shape_count:])
                )
                if global_shape_count
                else source[:shapes_per_world]
            )
            self._shared_shape_arrays[native_name] = wp.array(
                compact, dtype=dtype, device=self.model.device
            )

    @property
    def synreal_model(self) -> Model:
        """Return the native model containing non-owning Newton array views."""
        return self._synreal_model

    @property
    def native_solver(self) -> Solver:
        """Return the wrapped native solver."""
        return self._native_solver

    @property
    def config(self) -> Config:
        """Return the mutable native configuration used by subsequent steps."""
        return self._native_solver.config

    def prepare_contacts(self, contacts: Any | None) -> None:
        """Reserve native hard-contact storage for a collision buffer before capture.

        Use the actual pipeline buffer, including proxy-local buffers in coupled
        simulations: their capacity can exceed the model's initial estimate.
        Repeated calls with an already prepared capacity are capture-safe.
        """
        if contacts is None:
            return
        self._reserve_rigid_contact_capacity(int(contacts.rigid_contact_max))

    def _reserve_rigid_contact_capacity(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("Newton rigid contact capacity cannot be negative.")
        if capacity <= self._rigid_contact_capacity:
            return
        if self._wp.get_stream(self.device).is_capturing:
            raise RuntimeError(
                "Call NewtonSolver.prepare_contacts(contacts) before CUDA graph capture: "
                f"contact capacity {capacity} exceeds the reserved capacity {self._rigid_contact_capacity}."
            )
        with self._wp.ScopedDevice(self.device):
            self._native_solver.reserve_rigid_contact_capacity(capacity)
        self._rigid_contact_capacity = capacity

    def enable_joint_drive_feedback(self) -> Any:
        """Enable recording of the pre-integration motor effort for each step.

        Call before CUDA graph capture. The independent CUDA output array does
        not add State fields or change the state, parameters, or dynamics solve.
        """
        if self._joint_drive_effort is None:
            if self._wp.get_stream(self.device).is_capturing:
                raise RuntimeError("Enable joint drive feedback before CUDA graph capture.")
            count = int(self.model.joint_dof_count)
            self._joint_drive_effort = self._wp.zeros(count, dtype=self._wp.float32, device=self.device)
        self._ensure_joint_drive_feedback_angle_state()
        return self._joint_drive_effort

    def _ensure_joint_drive_feedback_angle_state(self) -> None:
        """Allocate the persistent local angle branch used by feedback sampling."""
        drive_count = 0 if self._joint_drive_types is None else len(self._joint_drive_types)
        dof_count = int(self.model.joint_dof_count)
        if (
            self._joint_drive_feedback_previous_angles is not None
            and self._joint_drive_feedback_angle_initialized is not None
            and len(self._joint_drive_feedback_previous_angles) == drive_count
            and len(self._joint_drive_feedback_angle_initialized) == drive_count
            and self._joint_drive_feedback_scratch is not None
            and len(self._joint_drive_feedback_scratch) == dof_count
            and self._joint_drive_start_velocities is not None
            and len(self._joint_drive_start_velocities) == drive_count
        ):
            return
        if self._wp.get_stream(self.device).is_capturing:
            raise RuntimeError("Initialize joint-drive feedback state before CUDA graph capture.")
        self._joint_drive_feedback_previous_angles = self._wp.zeros(
            drive_count, dtype=self._wp.float32, device=self.device
        )
        self._joint_drive_feedback_angle_initialized = self._wp.zeros(
            drive_count, dtype=self._wp.int32, device=self.device
        )
        self._joint_drive_reset_mask = self._wp.empty(drive_count, dtype=self._wp.int32, device=self.device)
        self._joint_drive_feedback_scratch = self._wp.empty(
            dof_count, dtype=self._wp.float32, device=self.device
        )
        self._joint_drive_start_velocities = self._wp.zeros(
            drive_count, dtype=self._wp.float32, device=self.device
        )

    def disable_joint_drive_feedback(self) -> None:
        """Disable feedback; discard captured graphs referencing the output before calling.

        Existing CUDA graphs keep their captured output address and must be rebuilt
        after enabling or disabling feedback, just as after configuration changes.
        """
        if self._wp.get_stream(self.device).is_capturing:
            raise RuntimeError("Disable joint drive feedback outside CUDA graph capture.")
        self._joint_drive_effort = None
        self._joint_drive_feedback_state = None
        self._joint_drive_feedback_previous_state = None
        self._joint_drive_feedback_dt = None

    @property
    def joint_drive_effort(self) -> Any | None:
        """Return the optional motor-effort CUDA array, indexed like ``control.joint_f``.

        Values are signed along each DOF axis: N for prismatic drives and Nm for
        angular drives. Values are ``joint_f`` plus the mode-aware, clipped
        actuator effort at the beginning of the step, matching MuJoCo's force
        evaluation. They exclude passive damping, contact, and joint reactions. Unsupported DOFs contain
        zero. Copy after each step to retain history; CUDA graph replay also
        overwrites this independent array. No State fields are added or changed.
        """
        return self._joint_drive_effort

    def compute_joint_drive_effort(
        self,
        state: Any,
        control: Any | None,
        out: Any | None = None,
        *,
        previous_state: Any | None = None,
        dt: float | None = None,
        _advance_angle_state: bool = False,
    ) -> Any:
        """Evaluate the supplied state's instantaneous motor effort without changing it.

        Velocity comes from body twists, while :meth:`step` records this value
        before integration. ``previous_state`` and ``dt`` remain accepted for
        compatibility but do not replace the instantaneous velocity with a secant.
        A supplied previous pose seeds the angular branch when it is uninitialized.
        Pass a preallocated float32 ``out`` array during CUDA graph capture.
        """
        count = int(self.model.joint_dof_count)
        stream = self._wp.get_stream(self.device)
        if out is None:
            if stream.is_capturing:
                raise RuntimeError("Preallocate the motor-effort output before CUDA graph capture.")
            out = self._wp.empty(count, dtype=self._wp.float32, device=self.device)
        if tuple(out.shape) != (count,):
            raise ValueError(f"Motor-effort output must have shape ({count},).")
        self._as_span(out, count, expected_dtype=self._wp.float32)
        self._validate_body_state_arrays(state)
        if previous_state is not None:
            self._validate_body_state_arrays(previous_state)
            if dt is None or not np.isfinite(dt) or float(dt) <= 0.0:
                raise ValueError(
                    "dt must be finite and greater than zero when previous_state is provided."
                )
        # Clearing an aliased buffer could silently alter the state or the held motor command.
        for owner in (state, previous_state, control, self.model):
            if owner is None:
                continue
            for array in getattr(owner, "__dict__", {}).values():
                if (isinstance(array, self._wp.array) and out.ptr and array.ptr
                        and out.ptr < array.ptr + array.capacity and array.ptr < out.ptr + out.capacity):
                    raise ValueError("Motor-effort output must not alias state, controls, or motor parameters.")
        if self._joint_drive_types is None or not len(self._joint_drive_types):
            out.zero_()
            return out
        self._ensure_joint_drive_feedback_angle_state()
        if control is None:
            raise ValueError("A Newton control is required for driven joints.")
        self._as_synreal_control(control)
        out.zero_()
        self._wp.launch(
            evaluate_joint_drive_effort,
            dim=len(self._joint_drive_types),
            inputs=[
                state.body_q,
                state.body_q if previous_state is None else previous_state.body_q,
                self._joint_drive_feedback_previous_angles,
                self._joint_drive_feedback_angle_initialized,
                state.body_qd, self.model.body_com, self._affine_body_ids,
                self._joint_drive_types, self._joint_drive_body_indices,
                self._joint_drive_parent_origins, self._joint_drive_child_origins,
                self._joint_drive_parent_axis_points, self._joint_drive_parent_reference_points,
                self._joint_drive_child_reference_points, self._joint_drive_dof_indices,
                self._joint_drive_target_indices, self.model.joint_target_ke, self.model.joint_target_kd,
                self.model.joint_target_mode,
                self.model.joint_effort_limit, control.joint_f, control.joint_target_q, control.joint_target_qd,
                int(previous_state is not None),
                int(_advance_angle_state),
            ],
            outputs=[out, self._joint_drive_start_velocities], device=self.device, stream=stream,
        )
        return out

    def _find_active_joint_drive_body_ids(self) -> set[int]:
        """Return Newton body ids touched by a supported, active joint drive."""
        model = self.model
        if not getattr(model, "joint_count", 0):
            return set()

        joint_types = model.joint_type.numpy()
        joint_parents = model.joint_parent.numpy()
        joint_children = model.joint_child.numpy()
        joint_dof_dims = model.joint_dof_dim.numpy()
        joint_dof_starts = model.joint_qd_start.numpy()
        joint_enabled = model.joint_enabled.numpy()
        target_stiffnesses = np.asarray(model.joint_target_ke.numpy())
        target_dampings = np.asarray(model.joint_target_kd.numpy())
        active_bodies: set[int] = set()

        for joint in range(int(model.joint_count)):
            if not bool(joint_enabled[joint]):
                continue
            joint_type = int(joint_types[joint])
            linear_dofs, angular_dofs = (int(value) for value in joint_dof_dims[joint])
            dof_start = int(joint_dof_starts[joint])
            driven_dofs: tuple[int, ...] = ()
            if joint_type == int(_newton.JointType.REVOLUTE):
                driven_dofs = (dof_start,)
            elif joint_type == int(_newton.JointType.PRISMATIC):
                driven_dofs = (dof_start,)
            elif joint_type == int(_newton.JointType.D6):
                if (linear_dofs, angular_dofs) in ((0, 1), (1, 0)):
                    driven_dofs = (dof_start,)
                elif (linear_dofs, angular_dofs) == (1, 1):
                    driven_dofs = (dof_start, dof_start + 1)

            if not any(
                target_stiffnesses[dof] != 0.0 or target_dampings[dof] != 0.0
                for dof in driven_dofs
            ):
                continue
            parent = int(joint_parents[joint])
            child = int(joint_children[joint])
            if parent >= 0:
                active_bodies.add(parent)
            if child >= 0:
                active_bodies.add(child)

        return active_bodies

    def _ensure_affine_body_proxies(
        self,
        *,
        armature: float,
        armature_body_ids: set[int],
    ) -> None:
        """Append affine proxies while preserving finalized deformable particles."""
        model = self.model
        if (
            getattr(model, "synreal_affine_body_ids", None) is not None
            or not model.body_count
        ):
            return
        from .newton_model_builder import ModelBuilder

        body_flags = model.body_flags.numpy()
        body_worlds = model.body_world.numpy()
        body_transforms = model.body_q.numpy()
        body_velocities = model.body_qd.numpy()
        body_masses = model.body_mass.numpy()
        body_inertias = model.body_inertia.numpy()
        body_centers_of_mass = model.body_com.numpy()
        proxy_flags = int(_newton.BodyFlags.DYNAMIC | _newton.BodyFlags.KINEMATIC)
        body_ids = [
            body for body, flags in enumerate(body_flags) if int(flags) & proxy_flags
        ]
        proxy_worlds = [int(body_worlds[body]) for body in body_ids]

        world_count = int(model.world_count)
        if any(world < -1 or world >= world_count for world in proxy_worlds):
            raise ValueError(
                "A dynamic body's world index is outside the model world range."
            )
        original_particle_count = int(model.particle_count)
        positions: list[Any] = list(np.asarray(model.particle_q.numpy()))
        velocities: list[Any] = list(np.asarray(model.particle_qd.numpy()))
        particle_masses: list[float] = list(
            np.asarray(model.particle_mass.numpy(), dtype=np.float32)
        )
        particle_radii: list[float] = list(
            np.asarray(model.particle_radius.numpy(), dtype=np.float32)
        )
        particle_flags: list[int] = list(
            np.asarray(model.particle_flags.numpy(), dtype=np.int32)
        )
        particle_worlds: list[int] = list(
            np.asarray(model.particle_world.numpy(), dtype=np.int32)
        )
        particle_colors: list[int] = list(
            np.asarray(model.particle_colors.numpy(), dtype=np.int32)
        )
        affine_indices: list[tuple[int, int, int, int]] = []
        mass_matrices: list[Any] = []
        proxy_inertias: list[Any] = []

        for body in body_ids:
            mass = float(body_masses[body])
            if not np.isfinite(mass) or mass < 0.0:
                raise ValueError(
                    f"Dynamic body {body} must have non-negative mass before creating affine proxies."
                )
            # A prescribed endpoint remains represented in the cable topology,
            # but the inactive proxy's mass is only a nonsingular placeholder.
            proxy_mass = mass if mass > 0.0 else 1.0
            proxy_inertia = np.asarray(body_inertias[body], dtype=np.float64) if mass > 0.0 else np.eye(3)
            if body in armature_body_ids:
                # Generate both proxy geometry and coupled mass from the same
                # regularized inertia. Keeping these consistent gives driven
                # articulations substantially better nonlinear convergence.
                proxy_inertia = proxy_inertia + armature * np.eye(3)
            mass_matrix = ModelBuilder._proxy_mass_matrix(body, proxy_mass, proxy_inertia)
            transform = np.asarray(body_transforms[body], dtype=np.float64)
            rotation = np.asarray(
                wp.quat_to_matrix(wp.quat(*transform[3:7])), dtype=np.float64
            ).reshape(3, 3)
            world_com = transform[:3] + rotation @ np.asarray(
                body_centers_of_mass[body], dtype=np.float64
            )
            affine_columns = rotation.T.copy()
            body_velocity = np.asarray(body_velocities[body], dtype=np.float64)
            body_particle_positions = np.vstack((world_com, affine_columns))
            body_particle_velocities = np.vstack(
                (
                    body_velocity[:3],
                    [np.cross(body_velocity[3:], column) for column in affine_columns],
                )
            )

            first_particle = len(positions)
            indices = tuple(range(first_particle, first_particle + 4))
            affine_indices.append(indices)
            mass_matrices.append(mass_matrix)
            proxy_inertias.append(proxy_inertia)
            positions.extend(body_particle_positions)
            velocities.extend(body_particle_velocities)
            particle_masses.extend(np.diag(mass_matrix))
            particle_radii.extend((0.0,) * 4)
            particle_flags.extend((int(_newton.ParticleFlags.ACTIVE),) * 4)
            particle_worlds.extend((int(body_worlds[body]),) * 4)
            particle_colors.extend((0,) * 4)

        # Newton finalizes replicated deformable particles as one contiguous block
        # per world, while affine proxies are discovered here after finalization.
        # Restore the required world-local state layout before exposing either set
        # of indices to Newton or the native replicated solver.
        if world_count > 1 and original_particle_count:
            final_order = np.argsort(
                np.asarray(particle_worlds, dtype=np.int32), kind="stable"
            )
            if not np.array_equal(final_order, np.arange(len(particle_worlds))):
                old_to_new = np.empty(len(final_order), dtype=np.int32)
                old_to_new[final_order] = np.arange(len(final_order), dtype=np.int32)

                positions = [positions[index] for index in final_order]
                velocities = [velocities[index] for index in final_order]
                particle_masses = [particle_masses[index] for index in final_order]
                particle_radii = [particle_radii[index] for index in final_order]
                particle_flags = [particle_flags[index] for index in final_order]
                particle_worlds = [particle_worlds[index] for index in final_order]
                particle_colors = [particle_colors[index] for index in final_order]
                affine_indices = [
                    tuple(int(old_to_new[index]) for index in indices)
                    for indices in affine_indices
                ]

                def remap_particle_indices(name: str) -> None:
                    source = getattr(model, name, None)
                    if source is None:
                        return
                    values = np.asarray(source.numpy(), dtype=np.int32)
                    remapped = values.copy()
                    valid = remapped >= 0
                    remapped[valid] = old_to_new[remapped[valid]]
                    setattr(
                        model,
                        name,
                        wp.array(remapped, dtype=source.dtype, device=model.device),
                    )

                remap_particle_indices("spring_indices")
                remap_particle_indices("tri_indices")
                remap_particle_indices("edge_indices")
                remap_particle_indices("tet_indices")
                remap_particle_indices("synreal_fluid_particle_indices")
                remap_particle_indices("synreal_granular_particle_indices")
                remap_particle_indices("synreal_granular_bond_indices")

                if model.particle_color_groups:
                    model.particle_color_groups = [
                        wp.array(
                            old_to_new[np.asarray(group.numpy(), dtype=np.int32)],
                            dtype=wp.int32,
                            device=model.device,
                        )
                        for group in model.particle_color_groups
                    ]
        particle_count = len(positions)
        device = model.device
        requires_grad = bool(model.requires_grad)
        masses_np = np.asarray(particle_masses, dtype=np.float32)

        model.particle_count = particle_count
        model.particle_q = wp.array(
            positions, dtype=wp.vec3, device=device, requires_grad=requires_grad
        )
        model.particle_qd = wp.array(
            velocities, dtype=wp.vec3, device=device, requires_grad=requires_grad
        )
        model.particle_mass = wp.array(
            masses_np, dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        inverse_masses = np.zeros_like(masses_np)
        np.divide(1.0, masses_np, out=inverse_masses, where=masses_np > 0.0)
        model.particle_inv_mass = wp.array(
            inverse_masses,
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        model.particle_radius = wp.array(
            particle_radii,
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        model.particle_flags = wp.array(particle_flags, dtype=wp.int32, device=device)
        model.particle_world = wp.array(particle_worlds, dtype=wp.int32, device=device)
        ordered_worlds = [-1 if world == -1 else world for world in particle_worlds]
        if any(
            first > second for first, second in zip(ordered_worlds, ordered_worlds[1:])
        ):
            raise ValueError(
                "Existing particles and appended affine proxies are not grouped in "
                "Newton world order. Mixed deformable-rigid adaptation currently "
                "requires their particles to occupy the same final world block."
            )
        world_starts = np.zeros(world_count + 2, dtype=np.int32)
        world_starts[0] = sum(world == -1 for world in particle_worlds)
        world_histogram = np.bincount(
            np.asarray(
                [world for world in particle_worlds if world >= 0], dtype=np.int64
            ),
            minlength=world_count,
        )
        for world in range(world_count):
            world_starts[world + 1] = world_starts[world] + world_histogram[world]
        world_starts[-1] = particle_count
        model.particle_world_start = wp.array(
            world_starts, dtype=wp.int32, device=device
        )
        model.particle_colors = wp.array(particle_colors, dtype=wp.int32, device=device)
        if original_particle_count == 0:
            model.particle_color_groups = [
                wp.array(range(particle_count), dtype=wp.int32, device=device)
            ]
        model.particle_grid = None

        mass_diagonal = masses_np.copy()
        non_diagonal_indices: list[tuple[int, int]] = []
        non_diagonal_values: list[float] = []
        for indices, mass_matrix in zip(affine_indices, mass_matrices):
            for local_index, particle in enumerate(indices):
                mass_diagonal[particle] = mass_matrix[local_index, local_index]
            for first, second in combinations(range(4), 2):
                value = float(mass_matrix[first, second])
                if value != 0.0:
                    non_diagonal_indices.append((indices[first], indices[second]))
                    non_diagonal_values.append(value)

        object.__setattr__(model, "synreal_particle_mass_diagonal", wp.array(
            mass_diagonal, dtype=wp.float32, device=device
        ))
        object.__setattr__(model, "synreal_particle_mass_non_diagonal_indices", wp.array(
            non_diagonal_indices, dtype=wp.vec2i, device=device
        ))
        object.__setattr__(model, "synreal_particle_mass_non_diagonal_values", wp.array(
            non_diagonal_values, dtype=wp.float32, device=device
        ))
        object.__setattr__(model, "synreal_affine_body_ids", wp.array(
            body_ids, dtype=wp.int32, device=device
        ))
        object.__setattr__(model, "synreal_affine_body_indices", wp.array(
            affine_indices, dtype=wp.vec4i, device=device
        ))
        object.__setattr__(model, "synreal_affine_body_masses", wp.array(
            np.where(body_masses[body_ids] > 0.0, body_masses[body_ids], 1.0), dtype=wp.float32, device=device
        ))
        object.__setattr__(model, "synreal_affine_body_inertia", wp.array(
            proxy_inertias, dtype=wp.mat33, device=device
        ))
        object.__setattr__(model, "synreal_affine_body_com", wp.array(
            body_centers_of_mass[body_ids], dtype=wp.vec3, device=device
        ))
        self._pre_regularized_body_ids = set(body_ids) & armature_body_ids

    @staticmethod
    def _joint_frame_points(transform: Any, offsets: list[Any]) -> list[Any]:
        """Transform joint-frame offsets into a body-local or world frame."""
        values = np.asarray(transform, dtype=np.float64)
        position = values[:3]
        rotation = np.asarray(
            wp.quat_to_matrix(wp.quat(*values[3:7])), dtype=np.float64
        ).reshape(3, 3)
        return [position + rotation @ np.asarray(offset) for offset in offsets]

    @staticmethod
    def _perpendicular_axes(axis: Any) -> tuple[Any, Any]:
        """Return a stable orthonormal basis perpendicular to an axis."""
        axis = np.asarray(axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        reference = np.zeros(3, dtype=np.float64)
        reference[int(np.argmin(np.abs(axis)))] = 1.0
        first = np.cross(axis, reference)
        first /= np.linalg.norm(first)
        return first, np.cross(axis, first)

    def _assemble_cable_constraints(self) -> None:
        """Preserve Newton's four cable material slots and DER rest invariants."""
        model = self.model
        joint_types = model.joint_type.numpy()
        joints = np.flatnonzero(joint_types == int(_newton.JointType.ROD)).astype(np.int32)
        self._cable_count = len(joints)
        self._cable_joints = wp.array(joints, dtype=wp.int32, device=self.device)
        if not self._cable_count:
            return

        affine_by_body = {int(body): affine for affine, body in enumerate(self._affine_body_ids.numpy())}
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        body_indices = []
        for joint in joints:
            parent, child = int(parents[joint]), int(children[joint])
            if child not in affine_by_body or (parent >= 0 and parent not in affine_by_body):
                raise NotImplementedError(f"Cable joint {joint} requires affine proxies for both endpoints.")
            body_indices.append((-1 if parent < 0 else affine_by_body[parent], affine_by_body[child]))
        starts = model.joint_qd_start.numpy()[joints]
        stiffness = model.joint_target_ke.numpy()
        damping = model.joint_target_kd.numpy()
        self._cable_body_indices = wp.array(body_indices, dtype=wp.vec2i, device=self.device)
        self._cable_parent_transforms = wp.array(model.joint_X_p.numpy()[joints], dtype=wp.transform, device=self.device)
        self._cable_child_transforms = wp.array(model.joint_X_c.numpy()[joints], dtype=wp.transform, device=self.device)
        self._cable_stiffnesses = wp.array([stiffness[start:start + 4] for start in starts], dtype=wp.vec4, device=self.device)
        self._cable_dampings = wp.array([damping[start:start + 4] for start in starts], dtype=wp.vec4, device=self.device)
        for name in ("_cable_rest", "_cable_bias", "_cable_hessian"):
            setattr(self, name, wp.zeros(self._cable_count, dtype=wp.vec3, device=self.device))
        self._cable_enabled = wp.zeros(self._cable_count, dtype=wp.int32, device=self.device)
        self._cable_penalty_reset_mask = wp.zeros(self._cable_count, dtype=wp.int32, device=self.device)

        joint_count = int(model.joint_count)
        self._joint_cable_rest_kb = wp.zeros(joint_count, dtype=wp.vec3, device=self.device)
        self._joint_cable_rest_twist = wp.zeros(joint_count, dtype=wp.float32, device=self.device)
        self._refresh_cable_rest_bend_twist()
        for name in ("joint_sigma_prev", "joint_kappa_prev", "_joint_dkappa_prev", "_joint_sigma_start", "_joint_C_fric"):
            setattr(self, name, wp.zeros(joint_count, dtype=wp.vec3, device=self.device))
        self._cable_soft_slots = wp.zeros(int(model.joint_dof_count), dtype=wp.int32, device=self.device)
        vbd_attributes = getattr(model, "vbd", None)
        self._cable_eps_max = getattr(vbd_attributes, "dahl_eps_max", None)
        self._cable_tau = getattr(vbd_attributes, "dahl_tau", None)
        if self._cable_eps_max is None:
            self._cable_eps_max = wp.zeros(joint_count, dtype=wp.float32, device=self.device)
        if self._cable_tau is None:
            self._cable_tau = wp.zeros(joint_count, dtype=wp.float32, device=self.device)
        self._cable_has_dahl = bool(np.any((self._cable_eps_max.numpy() > 0.0) & (self._cable_tau.numpy() > 0.0)))
        self._cable_rebaseline_mask = wp.ones(int(model.world_count) + 1, dtype=wp.bool, device=self.device)
        self._gather_cable_parameters()

    def _refresh_cable_rest_bend_twist(self) -> None:
        """Recompute DER rest invariants after rest poses or joint frames change."""
        if not self._cable_count:
            return
        model = self.model
        wp.launch(
            init_cable_rest_bend_twist,
            dim=int(model.joint_count),
            inputs=[model.joint_type, model.joint_parent, model.joint_child, model.joint_X_p, model.joint_X_c, model.body_q],
            outputs=[self._joint_cable_rest_kb, self._joint_cable_rest_twist],
            device=self.device,
        )

    def _gather_cable_parameters(self) -> None:
        wp.launch(
            _gather_cable_parameters,
            dim=self._cable_count,
            inputs=[self._cable_joints, self.model.joint_enabled, self._joint_cable_rest_kb,
                    self._joint_cable_rest_twist, self._joint_sigma_start, self._joint_C_fric],
            outputs=[self._cable_enabled, self._cable_rest, self._cable_bias, self._cable_hessian],
            device=self.device,
        )

    def _prepare_cable_dahl(self, state: Any) -> None:
        model = self.model
        if self._cable_has_dahl:
            wp.launch(
                compute_cable_dahl_parameters,
                dim=int(model.joint_count),
                inputs=[model.joint_type, model.joint_enabled, model.joint_world, self._cable_rebaseline_mask,
                        model.joint_parent, model.joint_child, model.joint_X_p, model.joint_X_c, model.joint_qd_start,
                        model.joint_target_ke, self._cable_soft_slots, self._joint_cable_rest_kb,
                        self._joint_cable_rest_twist, state.body_q, self.joint_sigma_prev, self.joint_kappa_prev,
                        self._joint_dkappa_prev, self._cable_eps_max, self._cable_tau],
                outputs=[self._joint_sigma_start, self._joint_C_fric],
                device=self.device,
            )
            self._cable_rebaseline_mask.zero_()
        self._gather_cable_parameters()

    def _update_cable_dahl(self, state: Any) -> None:
        if not self._cable_has_dahl:
            return
        model = self.model
        wp.launch(
            update_cable_dahl_state,
            dim=int(model.joint_count),
            inputs=[model.joint_type, model.joint_enabled, model.joint_parent, model.joint_child, model.joint_X_p,
                    model.joint_X_c, model.joint_qd_start, model.joint_target_ke, self._cable_soft_slots,
                    self._joint_cable_rest_kb, self._joint_cable_rest_twist, state.body_q,
                    self._cable_eps_max, self._cable_tau],
            outputs=[self.joint_sigma_prev, self.joint_kappa_prev, self._joint_dkappa_prev],
            device=self.device,
        )

    def _prepare_affine_inertial_refresh(self) -> None:
        """Reserve all mutable column-mass entries before native sparsity is built."""
        self._affine_mass_pair_slots = None
        self._affine_inertia_offsets = None
        self._affine_refresh_com = None
        self._affine_refresh_linear_mobility = None
        self._affine_refresh_angular_mobility = None
        if self._affine_body_ids is None or not len(self._affine_body_ids):
            return
        body_ids = np.asarray(self._affine_body_ids.numpy(), dtype=np.int64)
        self._affine_refresh_com = self.model.body_com.numpy()[body_ids].copy()
        dynamic = (self.model.body_flags.numpy()[body_ids] & int(_newton.BodyFlags.KINEMATIC)) == 0
        self._affine_refresh_linear_mobility = dynamic & (self.model.body_inv_mass.numpy()[body_ids] != 0.0)
        self._affine_refresh_angular_mobility = dynamic & np.any(
            self.model.body_inv_inertia.numpy()[body_ids] != 0.0, axis=(1, 2)
        )
        indices = np.asarray(self._affine_body_indices.numpy(), dtype=np.int64)
        source_pairs = (
            np.empty((0, 2), dtype=np.int64)
            if self._particle_mass_non_diagonal_indices is None
            else np.asarray(self._particle_mass_non_diagonal_indices.numpy(), dtype=np.int64)
        )
        source_values = (
            np.empty(0, dtype=np.float32)
            if self._particle_mass_non_diagonal_values is None
            else np.asarray(self._particle_mass_non_diagonal_values.numpy(), dtype=np.float32)
        )
        entries: dict[tuple[int, int], float] = {}
        for pair, value in zip(source_pairs, source_values):
            key = tuple(sorted((int(pair[0]), int(pair[1]))))
            entries[key] = entries.get(key, 0.0) + float(value)
        column_pairs = []
        for particles in indices:
            pairs = [tuple(sorted((int(particles[first]), int(particles[second]))))
                     for first, second in ((1, 2), (1, 3), (2, 3))]
            column_pairs.append(pairs)
            # A zero currently has no force, but its slot must exist when ADMM
            # later changes an off-diagonal inertia during graph replay.
            for pair in pairs:
                entries.setdefault(pair, 0.0)
        ordered = sorted(entries)
        slot_by_pair = {pair: slot for slot, pair in enumerate(ordered)}
        device = self.device
        self._particle_mass_non_diagonal_indices = wp.array(ordered, dtype=wp.vec2i, device=device)
        self._particle_mass_non_diagonal_values = wp.array(
            [entries[pair] for pair in ordered], dtype=wp.float32, device=device
        )
        self._affine_mass_pair_slots = wp.array(
            [[slot_by_pair[pair] for pair in pairs] for pairs in column_pairs], dtype=wp.vec3i, device=device
        )
        # Legacy joint-drive armature belongs to the solver metric, so preserve
        # that fixed offset when the coupler overwrites the physical inertia.
        authored = np.asarray(self.model.body_inertia.numpy(), dtype=np.float32)[body_ids]
        current = np.asarray(self._affine_body_inertia.numpy(), dtype=np.float32)
        offsets = current - authored
        fixed_flags = self._fixed_affine_body_flags.numpy()
        locked_flags = self._rotation_locked_affine_body_flags.numpy()
        for affine in range(len(body_ids)):
            if fixed_flags[affine] or locked_flags[affine]:
                offsets[affine] = 0.0
        self._affine_inertia_offsets = wp.array(offsets, dtype=wp.mat33, device=device)

    def _affine_inertial_refresh_preserves_layout(self) -> bool:
        """Check eager updates for changes requiring cached weights or masks to be rebuilt."""
        if self._affine_refresh_com is None:
            return True
        body_ids = np.asarray(self._affine_body_ids.numpy(), dtype=np.int64)
        dynamic = (self.model.body_flags.numpy()[body_ids] & int(_newton.BodyFlags.KINEMATIC)) == 0
        return (
            np.array_equal(self.model.body_com.numpy()[body_ids], self._affine_refresh_com)
            and np.array_equal(
                dynamic & (self.model.body_inv_mass.numpy()[body_ids] != 0.0), self._affine_refresh_linear_mobility
            )
            and np.array_equal(
                dynamic & np.any(self.model.body_inv_inertia.numpy()[body_ids] != 0.0, axis=(1, 2)),
                self._affine_refresh_angular_mobility,
            )
        )

    def _refresh_affine_mass_properties_device(self) -> None:
        """Update ADMM mass/inertia metrics without allocations or host reads."""
        if self._affine_mass_pair_slots is None:
            return
        wp.launch(
            _refresh_affine_body_mass_kernel,
            dim=len(self._affine_body_ids),
            inputs=[self.model.body_mass, self.model.body_inertia, self.model.body_com, self._affine_body_ids,
                    self._affine_body_indices, self._rotation_locked_affine_body_flags, self._affine_inertia_offsets,
                    self._affine_mass_pair_slots],
            outputs=[self._particle_mass_diagonal, self._particle_mass_non_diagonal_values, self._affine_body_masses,
                     self._affine_body_inertia, self._affine_body_com],
            device=self.device,
            stream=wp.get_stream(self.device),
        )

    def _refresh_affine_mass_properties(self) -> None:
        """Rebuild the coupled proxy metric after body inertia changes outside graph capture."""
        if self._affine_body_ids is None or not len(self._affine_body_ids):
            return
        from .newton_model_builder import ModelBuilder

        body_ids = np.asarray(self._affine_body_ids.numpy(), dtype=np.int64)
        affine_indices = np.asarray(self._affine_body_indices.numpy(), dtype=np.int64)
        body_mass = np.asarray(self.model.body_mass.numpy(), dtype=np.float64)[body_ids]
        body_inertia = np.asarray(self.model.body_inertia.numpy(), dtype=np.float64)[body_ids].reshape((-1, 3, 3))
        if not np.isfinite(body_mass).all() or np.any(body_mass < 0.0):
            raise ValueError("Affine body masses must remain finite and non-negative.")
        # Couplers can assign a positive proximal mass to a disabled body. Its
        # inverse mass, rather than the authored mass, determines whether it moves.
        proxy_mass = np.where(body_mass > 0.0, body_mass, 1.0)
        proxy_inertia = body_inertia.copy()
        proxy_inertia[body_mass == 0.0] = np.eye(3)
        body_flags = self.model.body_flags.numpy()[body_ids]
        rotation_locked = ((body_flags & int(_newton.BodyFlags.PROXY)) != 0) & np.all(body_inertia == 0.0, axis=(1, 2))
        self._rotation_locked_affine_body_ids = np.flatnonzero(rotation_locked).tolist()
        self._rotation_locked_affine_body_flags = wp.array(rotation_locked.astype(np.int32), device=self.device)
        # Coupled articulated proxies can carry translational effective mass and
        # zero angular mobility. Their inactive affine columns need a valid dummy
        # metric, while physical dynamic bodies retain the strict inertia checks.
        proxy_inertia[rotation_locked] = np.eye(3)
        driven_bodies = self._find_active_joint_drive_body_ids()
        for affine, body in enumerate(body_ids):
            if int(body) in driven_bodies:
                proxy_inertia[affine] += self._joint_drive_armature * np.eye(3)

        mass_diagonal = (
            self._particle_mass_diagonal if self._particle_mass_diagonal is not None else self.model.particle_mass
        ).numpy().copy()
        proxy_particles = set(affine_indices.reshape(-1).tolist())
        non_diagonal = {}
        if self._particle_mass_non_diagonal_indices is not None:
            for pair, value in zip(
                self._particle_mass_non_diagonal_indices.numpy(), self._particle_mass_non_diagonal_values.numpy()
            ):
                if int(pair[0]) not in proxy_particles and int(pair[1]) not in proxy_particles:
                    non_diagonal[tuple(int(index) for index in pair)] = float(value)
        for affine, indices in enumerate(affine_indices):
            mass_matrix = ModelBuilder._proxy_mass_matrix(int(body_ids[affine]), proxy_mass[affine], proxy_inertia[affine])
            mass_diagonal[indices] = np.diag(mass_matrix)
            for first, second in combinations(range(4), 2):
                value = float(mass_matrix[first, second])
                if value != 0.0:
                    non_diagonal[(int(indices[first]), int(indices[second]))] = value

        entries = sorted(non_diagonal.items())
        self._particle_mass_diagonal = wp.array(mass_diagonal, dtype=wp.float32, device=self.device)
        self._particle_mass_non_diagonal_indices = wp.array(
            [pair for pair, _ in entries], dtype=wp.vec2i, device=self.device
        )
        self._particle_mass_non_diagonal_values = wp.array(
            [value for _, value in entries], dtype=wp.float32, device=self.device
        )
        self._affine_body_masses = wp.array(proxy_mass, dtype=wp.float32, device=self.device)
        self._affine_body_inertia = wp.array(proxy_inertia, dtype=wp.mat33, device=self.device)
        self._affine_body_com = wp.array(self.model.body_com.numpy()[body_ids], dtype=wp.vec3, device=self.device)
        self._fixed_affine_body_ids = self._find_fixed_affine_bodies()
        fixed = set(self._fixed_affine_body_ids)
        self._fixed_affine_body_flags = wp.array(
            [int(affine in fixed) for affine in range(len(body_ids))], dtype=wp.int32, device=self.device
        )
        self._fixed_affine_bodies = (
            wp.array(self._fixed_affine_body_ids, dtype=wp.int32, device=self.device) if fixed else None
        )
        self._particle_flags = self._build_particle_flags()

    def _refresh_cable_materials(self) -> None:
        """Refresh mutable material slots while retaining DER rest shape and Dahl history."""
        if not self._cable_count:
            return
        joints = self._cable_joints.numpy()
        starts = self.model.joint_qd_start.numpy()[joints]
        stiffness = self.model.joint_target_ke.numpy()
        damping = self.model.joint_target_kd.numpy()
        self._cable_stiffnesses.assign(np.asarray([stiffness[start:start + 4] for start in starts]))
        self._cable_dampings.assign(np.asarray([damping[start:start + 4] for start in starts]))
        self._cable_parent_transforms.assign(self.model.joint_X_p.numpy()[joints])
        self._cable_child_transforms.assign(self.model.joint_X_c.numpy()[joints])
        self._cable_has_dahl = bool(np.any((self._cable_eps_max.numpy() > 0.0) & (self._cable_tau.numpy() > 0.0)))
        self._gather_cable_parameters()

    def _find_fixed_affine_bodies(self) -> list[int]:
        """Find prescribed affine bodies and bodies fixed to them or the world."""
        if self._affine_body_ids is None:
            return []

        affine_by_body = {
            int(body): affine_body
            for affine_body, body in enumerate(self._affine_body_ids.numpy())
        }
        joint_types = self.model.joint_type.numpy()
        joint_parents = self.model.joint_parent.numpy()
        joint_children = self.model.joint_child.numpy()
        joint_dof_dims = self.model.joint_dof_dim.numpy()
        joint_enabled = self.model.joint_enabled.numpy()
        inverse_mass = self.model.body_inv_mass.numpy()
        body_flags = self.model.body_flags.numpy()
        fixed_bodies = {
            body for body in affine_by_body
            if inverse_mass[body] == 0.0 or int(body_flags[body]) & int(_newton.BodyFlags.KINEMATIC)
        }

        changed = True
        while changed:
            changed = False
            for joint in range(int(self.model.joint_count)):
                if not bool(joint_enabled[joint]):
                    continue
                joint_type = int(joint_types[joint])
                is_fixed = joint_type == int(_newton.JointType.FIXED) or (
                    joint_type == int(_newton.JointType.D6)
                    and tuple(int(value) for value in joint_dof_dims[joint]) == (0, 0)
                )
                if not is_fixed:
                    continue

                parent = int(joint_parents[joint])
                child = int(joint_children[joint])
                if child not in affine_by_body or (
                    parent != -1 and parent not in fixed_bodies
                ):
                    continue
                if child not in fixed_bodies:
                    fixed_bodies.add(child)
                    changed = True

        return sorted(affine_by_body[body] for body in fixed_bodies)

    def _build_particle_flags(self) -> Any:
        """Return particle flags with fixed-body proxy particles made inactive."""
        if not self._fixed_affine_body_ids and not self._rotation_locked_affine_body_ids:
            return self.model.particle_flags

        particle_flags = self.model.particle_flags.numpy().copy()
        affine_body_indices = self._affine_body_indices.numpy()
        fixed_particle_indices = affine_body_indices[
            self._fixed_affine_body_ids
        ].reshape(-1)
        active_flag = int(_newton.ParticleFlags.ACTIVE)
        particle_flags[fixed_particle_indices] &= ~active_flag
        locked_columns = affine_body_indices[self._rotation_locked_affine_body_ids, 1:].reshape(-1)
        particle_flags[locked_columns] &= ~active_flag
        return wp.array(
            particle_flags,
            dtype=wp.int32,
            device=self.model.device,
        )

    def _build_collision_edge_indices(self) -> Any:
        """Keep every authored bending stencil but collide each physical edge only once."""
        indices = self.model.edge_indices
        if indices is None or not len(indices):
            return indices
        edges = np.asarray(indices.numpy(), dtype=np.int32)
        _, first = np.unique(np.sort(edges[:, 2:4], axis=1), axis=0, return_index=True)
        if len(first) == len(edges):
            return indices
        # A welded non-manifold seam can carry several sheet hinges with the same
        # endpoints. Repeating its contact primitive would multiply contact stiffness.
        return wp.array(edges[np.sort(first)], dtype=wp.int32, device=self.model.device)

    def _build_collision_particle_indices(self) -> Any:
        """Return active free particles with finite positive mass and radius."""
        particle_count = int(self.model.particle_count)
        constrained = np.zeros(particle_count, dtype=bool)

        def mark_particles(array: Any | None) -> None:
            if array is None or not len(array):
                return
            indices = np.asarray(array.numpy(), dtype=np.int64).reshape(-1)
            indices = indices[(indices >= 0) & (indices < particle_count)]
            constrained[indices] = True

        for topology in (
            self.model.spring_indices,
            self.model.tri_indices,
            self.model.edge_indices,
            self._tetrahedron_indices,
            self._affine_body_indices,
            self._particle_mass_non_diagonal_indices,
        ):
            mark_particles(topology)

        flags = np.asarray(self._particle_flags.numpy(), dtype=np.int32)
        masses = np.asarray(self.model.particle_mass.numpy(), dtype=np.float32)
        radii = np.asarray(self.model.particle_radius.numpy(), dtype=np.float32)
        active = (flags & int(_newton.ParticleFlags.ACTIVE)) != 0
        proxy = (flags & int(_newton.ParticleFlags.PROXY)) != 0
        free_particles = np.flatnonzero(
            ~constrained
            & active
            & ~proxy
            & np.isfinite(masses)
            & (masses > 0.0)
            & np.isfinite(radii)
            & (radii > 0.0)
        ).astype(np.int32)
        return wp.array(
            free_particles,
            dtype=wp.int32,
            device=self.model.device,
        )

    def _assemble_joint_constraints(
        self,
        *,
        damping: float,
        attachment_distance: float,
        line_half_length: float,
        drive_armature: float,
    ) -> None:
        """Translate supported Newton joints into affine point constraints."""
        if not getattr(self.model, "joint_count", 0):
            return

        affine_body_ids = (
            []
            if self._affine_body_ids is None
            else [int(body) for body in self._affine_body_ids.numpy()]
        )
        affine_by_body = {body: index for index, body in enumerate(affine_body_ids)}

        spring_indices: list[tuple[int, int]] = []
        spring_points0: list[Any] = []
        spring_points1: list[Any] = []
        spring_stiffnesses: list[float] = []
        spring_rest_lengths: list[float] = []
        line_indices: list[tuple[int, int]] = []
        line_points: list[Any] = []
        line_endpoints0: list[Any] = []
        line_endpoints1: list[Any] = []
        line_stiffnesses: list[float] = []
        line_dampings: list[float] = []
        drive_types: list[int] = []
        drive_body_indices: list[tuple[int, int]] = []
        drive_parent_origins: list[Any] = []
        drive_child_origins: list[Any] = []
        drive_parent_axis_points: list[Any] = []
        drive_parent_reference_points: list[Any] = []
        drive_child_reference_points: list[Any] = []
        drive_dof_indices: list[int] = []
        drive_target_indices: list[int] = []

        joint_types = self.model.joint_type.numpy()
        joint_parents = self.model.joint_parent.numpy()
        joint_children = self.model.joint_child.numpy()
        joint_parent_frames = self.model.joint_X_p.numpy()
        joint_child_frames = self.model.joint_X_c.numpy()
        joint_axes = self.model.joint_axis.numpy()
        joint_dof_dims = self.model.joint_dof_dim.numpy()
        joint_dof_starts = self.model.joint_qd_start.numpy()
        joint_target_starts = self.model.joint_target_q_start.numpy()
        joint_limit_lower = self.model.joint_limit_lower.numpy()
        joint_limit_upper = self.model.joint_limit_upper.numpy()
        joint_enabled = self.model.joint_enabled.numpy()

        def normalized_axis(value: Any, joint: int) -> Any:
            axis = np.asarray(value, dtype=np.float64)
            length = float(np.linalg.norm(axis))
            if not np.isfinite(length) or length <= 1.0e-8:
                raise ValueError(f"Joint {joint} has an invalid axis.")
            return axis / length

        for joint in range(int(self.model.joint_count)):
            if not bool(joint_enabled[joint]):
                continue

            if int(joint_types[joint]) == int(_newton.JointType.ROD):
                continue

            parent_body = int(joint_parents[joint])
            child_body = int(joint_children[joint])
            if child_body not in affine_by_body:
                raise NotImplementedError(
                    f"Joint {joint} child body {child_body} is not a dynamic affine body."
                )
            if parent_body != -1 and parent_body not in affine_by_body:
                raise NotImplementedError(
                    f"Joint {joint} parent body {parent_body} is not a dynamic affine body."
                )

            parent_affine = -1 if parent_body == -1 else affine_by_body[parent_body]
            child_affine = affine_by_body[child_body]
            if parent_affine == child_affine:
                raise ValueError(f"Joint {joint} connects a body to itself.")

            joint_type = int(joint_types[joint])
            linear_dofs, angular_dofs = (int(value) for value in joint_dof_dims[joint])
            axis_start = int(joint_dof_starts[joint])
            target_start = int(joint_target_starts[joint])
            linear_axes = [
                normalized_axis(joint_axes[axis_start + index], joint)
                for index in range(linear_dofs)
            ]
            linear_limits = [
                (
                    float(joint_limit_lower[axis_start + index]),
                    float(joint_limit_upper[axis_start + index]),
                )
                for index in range(linear_dofs)
            ]
            angular_axes = [
                normalized_axis(joint_axes[axis_start + linear_dofs + index], joint)
                for index in range(angular_dofs)
            ]

            construction: str
            axis: Any | None = None
            if joint_type == int(_newton.JointType.FREE):
                construction = "free"
            elif joint_type == int(_newton.JointType.FIXED):
                construction = "fixed"
            elif joint_type == int(_newton.JointType.BALL):
                construction = "ball"
            elif joint_type == int(_newton.JointType.REVOLUTE):
                construction, axis = "hinge", angular_axes[0]
            elif joint_type == int(_newton.JointType.PRISMATIC):
                construction, axis = "slider", linear_axes[0]
            elif joint_type == int(_newton.JointType.D6):
                if (linear_dofs, angular_dofs) == (3, 3):
                    construction = "free"
                elif (linear_dofs, angular_dofs) == (0, 0):
                    construction = "fixed"
                elif (linear_dofs, angular_dofs) == (0, 3):
                    construction = "ball"
                elif (linear_dofs, angular_dofs) == (0, 1):
                    construction, axis = "hinge", angular_axes[0]
                elif (linear_dofs, angular_dofs) == (1, 0):
                    construction, axis = "slider", linear_axes[0]
                elif (linear_dofs, angular_dofs) == (1, 1) and abs(
                    float(np.dot(linear_axes[0], angular_axes[0]))
                ) > 1.0 - 1.0e-5:
                    construction, axis = "cylindrical", linear_axes[0]
                else:
                    raise NotImplementedError(
                        f"Joint {joint} has unsupported D6 dimensions "
                        f"({linear_dofs} linear, {angular_dofs} angular)."
                    )
            else:
                raise NotImplementedError(
                    f"Joint {joint} type {joint_type} cannot be represented exactly "
                    "by point-point and point-line constraints."
                )

            drive_specs: list[tuple[int, Any, int, int]] = []
            if construction == "hinge":
                drive_specs.append((1, axis, axis_start, target_start))
            elif construction == "slider":
                drive_specs.append((0, axis, axis_start, target_start))
            elif construction == "cylindrical":
                drive_specs.extend(
                    (
                        (0, linear_axes[0], axis_start, target_start),
                        (1, angular_axes[0], axis_start + 1, target_start + 1),
                    )
                )

            for drive_type, drive_axis, dof_index, target_index in drive_specs:
                reference = np.zeros(3)
                if drive_type == 1:
                    reference, _ = self._perpendicular_axes(drive_axis)
                    reference = attachment_distance * reference

                parent_geometry = self._joint_frame_points(
                    joint_parent_frames[joint],
                    [np.zeros(3), drive_axis, reference],
                )
                child_geometry = self._joint_frame_points(
                    joint_child_frames[joint], [np.zeros(3), reference]
                )
                drive_types.append(drive_type)
                drive_body_indices.append((parent_affine, child_affine))
                drive_parent_origins.append(parent_geometry[0])
                drive_child_origins.append(child_geometry[0])
                drive_parent_axis_points.append(parent_geometry[1])
                drive_parent_reference_points.append(parent_geometry[2])
                drive_child_reference_points.append(child_geometry[1])
                drive_dof_indices.append(dof_index)
                drive_target_indices.append(target_index)

            if construction == "free":
                continue
            if construction == "fixed" and child_affine in self._fixed_affine_body_ids:
                continue

            if construction == "ball":
                offsets = [np.zeros(3)]
            elif construction == "hinge":
                offsets = [np.zeros(3), attachment_distance * axis]
            elif construction == "fixed":
                first = np.array((1.0, 0.0, 0.0))
                second = np.array((0.0, 1.0, 0.0))
                offsets = [
                    np.zeros(3),
                    attachment_distance * first,
                    attachment_distance * second,
                ]
            else:
                first, _ = self._perpendicular_axes(axis)
                offsets = [np.zeros(3), attachment_distance * axis]
                if construction == "slider":
                    offsets.extend(
                        [
                            attachment_distance * first,
                            attachment_distance * (first + axis),
                        ]
                    )

            parent_points = self._joint_frame_points(
                joint_parent_frames[joint], offsets
            )
            child_points = self._joint_frame_points(joint_child_frames[joint], offsets)
            if construction in ("ball", "hinge", "fixed"):
                for parent_point, child_point in zip(parent_points, child_points):
                    spring_indices.append((parent_affine, child_affine))
                    spring_points0.append(parent_point)
                    spring_points1.append(child_point)
                    spring_stiffnesses.append(1.0)
                    spring_rest_lengths.append(0.0)
                continue

            line_lower = -line_half_length
            line_upper = line_half_length
            if linear_limits:
                limit_lower, limit_upper = linear_limits[0]
                max_limit = float(_newton.MAXVAL)
                if np.isfinite(limit_lower) and limit_lower > -max_limit:
                    line_lower = limit_lower
                if np.isfinite(limit_upper) and limit_upper < max_limit:
                    line_upper = limit_upper
            if line_lower >= line_upper:
                raise ValueError(
                    f"Joint {joint} has an invalid linear limit interval "
                    f"[{line_lower}, {line_upper}]."
                )

            line_offsets = []
            point_offsets = offsets
            for point_offset in point_offsets:
                line_offsets.append(point_offset + line_lower * axis)
                line_offsets.append(point_offset + line_upper * axis)
            parent_line_points = self._joint_frame_points(
                joint_parent_frames[joint], line_offsets
            )
            constraint_damping = damping / len(child_points)
            for index, child_point in enumerate(child_points):
                line_indices.append((child_affine, parent_affine))
                line_points.append(child_point)
                line_endpoints0.append(parent_line_points[2 * index])
                line_endpoints1.append(parent_line_points[2 * index + 1])
                line_stiffnesses.append(1.0)
                line_dampings.append(constraint_damping)

        self._apply_joint_drive_armature(
            drive_body_indices,
            drive_dof_indices,
            drive_armature,
        )
        spring_stiffnesses, line_stiffnesses = (
            self._normalize_joint_constraint_stiffnesses(
                spring_indices,
                spring_points0,
                spring_points1,
                line_indices,
                line_points,
                line_endpoints0,
                line_endpoints1,
            )
        )

        def append_array(
            current: Any | None, generated: list[Any], dtype: Any
        ) -> Any | None:
            if not generated:
                return current
            values = np.asarray(generated)
            if current is not None and len(current):
                values = np.concatenate((current.numpy(), values), axis=0)
            return wp.array(values, dtype=dtype, device=self.model.device)

        self._body_spring_indices = append_array(
            self._body_spring_indices, spring_indices, wp.vec2i
        )
        self._body_spring_points0 = append_array(
            self._body_spring_points0, spring_points0, wp.vec3
        )
        self._body_spring_points1 = append_array(
            self._body_spring_points1, spring_points1, wp.vec3
        )
        self._body_spring_stiffnesses = append_array(
            self._body_spring_stiffnesses, spring_stiffnesses, wp.float32
        )
        self._body_spring_rest_lengths = append_array(
            self._body_spring_rest_lengths, spring_rest_lengths, wp.float32
        )
        self._body_point_line_indices = append_array(
            self._body_point_line_indices, line_indices, wp.vec2i
        )
        self._body_point_line_points = append_array(
            self._body_point_line_points, line_points, wp.vec3
        )
        self._body_point_line_endpoints0 = append_array(
            self._body_point_line_endpoints0, line_endpoints0, wp.vec3
        )
        self._body_point_line_endpoints1 = append_array(
            self._body_point_line_endpoints1, line_endpoints1, wp.vec3
        )
        self._body_point_line_stiffnesses = append_array(
            self._body_point_line_stiffnesses, line_stiffnesses, wp.float32
        )
        self._body_point_line_dampings = append_array(
            self._body_point_line_dampings, line_dampings, wp.float32
        )
        self._joint_body_spring_count = len(spring_indices)
        self._joint_body_point_line_count = len(line_indices)
        if drive_types:
            device = self.model.device
            self._joint_drive_types = wp.array(
                drive_types, dtype=wp.int32, device=device
            )
            self._joint_drive_body_indices = wp.array(
                drive_body_indices, dtype=wp.vec2i, device=device
            )
            self._joint_drive_parent_origins = wp.array(
                drive_parent_origins, dtype=wp.vec3, device=device
            )
            self._joint_drive_child_origins = wp.array(
                drive_child_origins, dtype=wp.vec3, device=device
            )
            self._joint_drive_parent_axis_points = wp.array(
                drive_parent_axis_points, dtype=wp.vec3, device=device
            )
            self._joint_drive_parent_reference_points = wp.array(
                drive_parent_reference_points, dtype=wp.vec3, device=device
            )
            self._joint_drive_child_reference_points = wp.array(
                drive_child_reference_points, dtype=wp.vec3, device=device
            )
            self._joint_drive_dof_indices = wp.array(
                drive_dof_indices, dtype=wp.int32, device=device
            )
            self._joint_drive_target_indices = wp.array(
                drive_target_indices, dtype=wp.int32, device=device
            )

    def _apply_joint_drive_armature(
        self,
        drive_body_indices: list[tuple[int, int]],
        drive_dof_indices: list[int],
        armature: float,
    ) -> None:
        """Regularize proxy inertia for bodies connected to active joint drives."""
        if armature == 0.0 or not drive_body_indices:
            return
        if (
            self._affine_body_ids is None
            or self._affine_body_indices is None
            or self._affine_body_masses is None
            or self._affine_body_inertia is None
        ):
            raise ValueError("Joint drive armature requires affine body metadata.")

        target_stiffnesses = np.asarray(
            self.model.joint_target_ke.numpy(), dtype=np.float64
        )
        target_dampings = np.asarray(
            self.model.joint_target_kd.numpy(), dtype=np.float64
        )
        driven_bodies: set[int] = set()
        for bodies, dof in zip(drive_body_indices, drive_dof_indices):
            if target_stiffnesses[dof] == 0.0 and target_dampings[dof] == 0.0:
                continue
            driven_bodies.update(body for body in bodies if body >= 0)
        affine_body_ids = np.asarray(self._affine_body_ids.numpy(), dtype=np.int64)
        driven_bodies = {
            body
            for body in driven_bodies
            if int(affine_body_ids[body]) not in self._pre_regularized_body_ids
        }
        if not driven_bodies:
            return

        from .newton_model_builder import ModelBuilder

        affine_indices = np.asarray(self._affine_body_indices.numpy(), dtype=np.int64)
        body_masses = np.asarray(self._affine_body_masses.numpy(), dtype=np.float64)
        body_inertias = np.asarray(
            self._affine_body_inertia.numpy(), dtype=np.float64
        ).reshape((-1, 3, 3))
        mass_diagonal_array = (
            self._particle_mass_diagonal
            if self._particle_mass_diagonal is not None
            else self.model.particle_mass
        )
        mass_diagonal = np.asarray(mass_diagonal_array.numpy(), dtype=np.float64)
        mass_pairs = (
            np.empty((0, 2), dtype=np.int64)
            if self._particle_mass_non_diagonal_indices is None
            else np.asarray(
                self._particle_mass_non_diagonal_indices.numpy(), dtype=np.int64
            )
        )
        mass_values = (
            np.empty(0, dtype=np.float64)
            if self._particle_mass_non_diagonal_values is None
            else np.asarray(
                self._particle_mass_non_diagonal_values.numpy(), dtype=np.float64
            )
        )
        non_diagonal = {
            tuple(sorted((int(pair[0]), int(pair[1])))): float(value)
            for pair, value in zip(mass_pairs, mass_values)
        }

        # Joint drive stiffness is authored independently of body inertia. Tiny
        # link inertias create angular modes that finite nonlinear/PCG iterations
        # cannot resolve reliably, so regularize only bodies touched by active
        # drives. Keep the existing proxy geometry and change its coupled mass;
        # this avoids moving particles when the solver is constructed.
        identity = np.eye(3, dtype=np.float64)
        for affine_body in sorted(driven_bodies):
            indices = affine_indices[affine_body]
            regularized_inertia = body_inertias[affine_body] + armature * identity
            mass_matrix = ModelBuilder._proxy_mass_matrix(
                int(affine_body_ids[affine_body]),
                float(body_masses[affine_body]),
                regularized_inertia,
            )
            body_inertias[affine_body] = regularized_inertia
            for local_index, particle in enumerate(indices):
                mass_diagonal[particle] = mass_matrix[local_index, local_index]
            for first, second in combinations(range(4), 2):
                pair = tuple(sorted((int(indices[first]), int(indices[second]))))
                value = float(mass_matrix[first, second])
                if value == 0.0:
                    non_diagonal.pop(pair, None)
                else:
                    non_diagonal[pair] = value

        device = self.model.device
        sorted_non_diagonal = sorted(non_diagonal.items())
        self._particle_mass_diagonal = wp.array(
            mass_diagonal, dtype=wp.float32, device=device
        )
        self._particle_mass_non_diagonal_indices = wp.array(
            [pair for pair, _ in sorted_non_diagonal], dtype=wp.vec2i, device=device
        )
        self._particle_mass_non_diagonal_values = wp.array(
            [value for _, value in sorted_non_diagonal], dtype=wp.float32, device=device
        )
        self._affine_body_inertia = wp.array(
            body_inertias, dtype=wp.mat33, device=device
        )

    def _normalize_body_distance_constraint_mass_scales(self) -> Any | None:
        """Build one coupled mass-normalized AL metric per hard constraint."""
        arrays = (
            self._body_distance_constraint_indices,
            self._body_distance_constraint_points0,
            self._body_distance_constraint_points1,
            self._body_distance_constraint_weights0,
            self._body_distance_constraint_weights1,
            self._body_distance_constraint_rest_lengths,
        )
        if not any(array is not None for array in arrays):
            return None
        if any(array is None for array in arrays):
            raise ValueError(
                "All body distance constraint arrays must be provided together."
            )
        constraint_count = len(self._body_distance_constraint_indices)
        if any(len(array) != constraint_count for array in arrays):
            raise ValueError("Body distance constraint arrays have different lengths.")
        if not constraint_count:
            return self._wp.array([], dtype=self._wp.float32, device=self.model.device)

        scales, _ = self._normalize_joint_constraint_stiffnesses(
            [
                (int(body0), int(body1))
                for body0, body1 in self._body_distance_constraint_indices.numpy()
            ],
            list(self._body_distance_constraint_points0.numpy()),
            list(self._body_distance_constraint_points1.numpy()),
            [],
            [],
            [],
            [],
            description="Body distance constraints",
        )
        return self._wp.array(scales, dtype=self._wp.float32, device=self.model.device)

    def _normalize_joint_constraint_stiffnesses(
        self,
        spring_indices: list[tuple[int, int]],
        spring_points0: list[Any],
        spring_points1: list[Any],
        line_indices: list[tuple[int, int]],
        line_points: list[Any],
        line_endpoints0: list[Any],
        line_endpoints1: list[Any],
        *,
        description: str = "Generated joints",
    ) -> tuple[list[float], list[float]]:
        """Mass-precondition and spectrally normalize generated joint metrics."""
        if not spring_indices and not line_indices:
            return [], []

        affine_indices = np.asarray(self._affine_body_indices.numpy(), dtype=np.int64)
        body_centers = np.asarray(self._affine_body_com.numpy(), dtype=np.float64)
        active_flag = int(_newton.ParticleFlags.ACTIVE)
        active = (
            np.asarray(self._particle_flags.numpy(), dtype=np.int64) & active_flag
        ) != 0

        def point_weights(body: int, point: Any) -> np.ndarray:
            offset = np.asarray(point, dtype=np.float64) - body_centers[body]
            return np.asarray(
                (1.0, offset[0], offset[1], offset[2]),
                dtype=np.float64,
            )

        def add_body_coefficients(
            coefficients: dict[int, float], body: int, point: Any, sign: float
        ) -> None:
            if body < 0:
                return
            for particle, weight in zip(
                affine_indices[body], point_weights(body, point)
            ):
                particle = int(particle)
                if active[particle]:
                    coefficients[particle] = coefficients.get(
                        particle, 0.0
                    ) + sign * float(weight)

        constraint_coefficients: list[dict[int, float]] = []
        for bodies, point0, point1 in zip(
            spring_indices, spring_points0, spring_points1
        ):
            coefficients: dict[int, float] = {}
            add_body_coefficients(coefficients, bodies[0], point0, -1.0)
            add_body_coefficients(coefficients, bodies[1], point1, 1.0)
            constraint_coefficients.append(coefficients)

        for bodies, point, endpoint0, endpoint1 in zip(
            line_indices, line_points, line_endpoints0, line_endpoints1
        ):
            coefficients = {}
            add_body_coefficients(coefficients, bodies[0], point, -1.0)
            if bodies[1] >= 0:
                midpoint = 0.5 * (
                    np.asarray(endpoint0, dtype=np.float64)
                    + np.asarray(endpoint1, dtype=np.float64)
                )
                add_body_coefficients(coefficients, bodies[1], midpoint, 1.0)
            constraint_coefficients.append(coefficients)

        particle_world = np.asarray(self.model.particle_world.numpy(), dtype=np.int64)
        constraints_by_world: dict[int, list[tuple[int, dict[int, float]]]] = {}
        for constraint, coefficients in enumerate(constraint_coefficients):
            if not coefficients:
                continue
            worlds = {int(particle_world[index]) for index in coefficients}
            if len(worlds) != 1:
                raise ValueError(f"{description} cross Newton worlds.")
            constraints_by_world.setdefault(worlds.pop(), []).append(
                (constraint, coefficients)
            )

        mass_diagonal_array = (
            self._particle_mass_diagonal
            if self._particle_mass_diagonal is not None
            else self.model.particle_mass
        )
        mass_diagonal = np.asarray(mass_diagonal_array.numpy(), dtype=np.float64)
        mass_pairs = (
            np.empty((0, 2), dtype=np.int64)
            if self._particle_mass_non_diagonal_indices is None
            else np.asarray(
                self._particle_mass_non_diagonal_indices.numpy(), dtype=np.int64
            )
        )
        mass_values = (
            np.empty(0, dtype=np.float64)
            if self._particle_mass_non_diagonal_values is None
            else np.asarray(
                self._particle_mass_non_diagonal_values.numpy(), dtype=np.float64
            )
        )

        constraint_stiffnesses = np.zeros(
            len(constraint_coefficients), dtype=np.float64
        )
        for constraints in constraints_by_world.values():
            particles = sorted({index for _, values in constraints for index in values})
            local_index = {particle: index for index, particle in enumerate(particles)}
            size = len(particles)
            mass = np.diag(mass_diagonal[particles])
            for pair, value in zip(mass_pairs, mass_values):
                first = local_index.get(int(pair[0]))
                second = local_index.get(int(pair[1]))
                if first is not None and second is not None:
                    mass[first, second] += value
                    mass[second, first] += value

            coefficient_vectors: list[tuple[int, np.ndarray]] = []
            for constraint, coefficients in constraints:
                vector = np.zeros(size, dtype=np.float64)
                for particle, coefficient in coefficients.items():
                    vector[local_index[particle]] = coefficient
                coefficient_vectors.append((constraint, vector))

            # Translation masses and rotational second moments have different
            # units and scales. Equilibrate before testing definiteness so a
            # heavy unrelated body cannot make a valid small link look singular.
            diagonal = np.diag(mass)
            if not np.isfinite(mass).all() or np.any(diagonal <= 0.0):
                raise ValueError(f"{description} reference a singular proxy mass matrix.")
            inverse_scale = np.reciprocal(np.sqrt(diagonal))
            scaled_mass = inverse_scale[:, None] * mass * inverse_scale[None, :]
            mass_values_eigen, mass_vectors = np.linalg.eigh(scaled_mass)
            tolerance = (
                32.0
                * np.finfo(np.float32).eps
                * max(float(np.max(mass_values_eigen)), np.finfo(np.float64).tiny)
            )
            if float(np.min(mass_values_eigen)) <= tolerance:
                raise ValueError(
                    f"{description} reference a singular proxy mass matrix."
                )
            # W.T @ M @ W = I; W need not be the symmetric inverse square root.
            mass_whitener = inverse_scale[:, None] * (
                mass_vectors * np.reciprocal(np.sqrt(mass_values_eigen))[None, :]
            )
            inverse_mass = mass_whitener @ mass_whitener.T

            # Give every scalar constraint unit maximum response in mass-normalized
            # coordinates. This prevents a light wrist constraint from setting the
            # stiffness of every heavier joint in the articulation.
            metric = np.zeros((size, size), dtype=np.float64)
            for constraint, vector in coefficient_vectors:
                response = float(vector @ inverse_mass @ vector)
                if not np.isfinite(response) or response <= 0.0:
                    raise ValueError(
                        f"{description} produced an invalid constraint metric."
                    )
                stiffness = 1.0 / response
                constraint_stiffnesses[constraint] = stiffness
                metric += stiffness * np.outer(vector, vector)

            # Coupled constraints can reinforce one another. A final per-world
            # Normalization preserves Config.constraint_alpha as the maximum
            # generalized eigenvalue of dt^2*K_joint against the proxy mass matrix.
            normalized_metric = mass_whitener.T @ metric @ mass_whitener
            maximum_eigenvalue = float(np.max(np.linalg.eigvalsh(normalized_metric)))
            if not np.isfinite(maximum_eigenvalue) or maximum_eigenvalue <= 0.0:
                raise ValueError(
                    f"{description} produced an invalid constraint metric."
                )
            for constraint, _ in coefficient_vectors:
                constraint_stiffnesses[constraint] /= maximum_eigenvalue

        spring_count = len(spring_indices)
        return (
            constraint_stiffnesses[:spring_count].tolist(),
            constraint_stiffnesses[spring_count:].tolist(),
        )

    def _validate_model(self) -> None:
        """Validate the subset of Newton models supported by the backend."""
        device = getattr(self.model, "device", None)
        if device is None or not device.is_cuda:
            raise ValueError(
                "NewtonSolver requires a Newton model allocated on a CUDA device."
            )

        mass_diagonal = (
            self._particle_mass_diagonal
            if self._particle_mass_diagonal is not None
            else self.model.particle_mass
        )
        if self.model.particle_count and mass_diagonal is None:
            raise ValueError("Newton model particle mass diagonal must be allocated.")
        if (
            mass_diagonal is not None
            and len(mass_diagonal) != self.model.particle_count
        ):
            raise ValueError(
                "Particle mass diagonal does not match the particle count."
            )

        tetrahedron_count = int(self.model.tet_count)
        tetrahedron_arrays = (
            (
                self._tetrahedron_indices,
                (tetrahedron_count, 4),
                self._wp.int32,
                "indices",
            ),
            (
                self._tetrahedron_inverse_dms,
                (tetrahedron_count,),
                self._wp.mat33,
                "inverse rest-pose matrices",
            ),
            (
                self._tetrahedron_activations,
                (tetrahedron_count,),
                self._wp.float32,
                "activations",
            ),
            (
                self._tetrahedron_materials,
                (tetrahedron_count, 3),
                self._wp.float32,
                "materials",
            ),
        )
        if tetrahedron_count:
            for array, expected_shape, expected_dtype, name in tetrahedron_arrays:
                if array is None:
                    raise ValueError(
                        f"Newton tetrahedron {name} must be allocated when tet_count is nonzero."
                    )
                if tuple(array.shape) != expected_shape:
                    raise ValueError(
                        f"Newton tetrahedron {name} must have shape {expected_shape}, "
                        f"but received {tuple(array.shape)}."
                    )
                if array.dtype != expected_dtype:
                    raise ValueError(
                        f"Newton tetrahedron {name} must use {expected_dtype}, "
                        f"but received {array.dtype}."
                    )

            tetrahedron_indices = np.asarray(
                self._tetrahedron_indices.numpy(), dtype=np.int64
            )
            if np.any(tetrahedron_indices < 0) or np.any(
                tetrahedron_indices >= int(self.model.particle_count)
            ):
                raise ValueError(
                    "Newton tetrahedron indices reference an invalid particle."
                )

            inverse_dms = np.asarray(
                self._tetrahedron_inverse_dms.numpy(), dtype=np.float64
            )
            if not np.isfinite(inverse_dms).all():
                raise ValueError(
                    "Newton tetrahedron inverse rest-pose matrices must be finite."
                )
            inverse_determinants = np.linalg.det(inverse_dms)
            if not np.isfinite(inverse_determinants).all() or np.any(
                inverse_determinants <= 0.0
            ):
                raise ValueError(
                    "Newton tetrahedron inverse rest-pose matrices must have positive determinants."
                )

            activations = np.asarray(
                self._tetrahedron_activations.numpy(), dtype=np.float64
            )
            if not np.isfinite(activations).all():
                raise ValueError("Newton tetrahedron activations must be finite.")

            materials = np.asarray(
                self._tetrahedron_materials.numpy(), dtype=np.float64
            )
            if not np.isfinite(materials).all() or np.any(materials < 0.0):
                raise ValueError(
                    "Newton tetrahedron material coefficients must be finite and non-negative."
                )

        if self._style3d is None:
            generic_surface_arrays = (
                (
                    int(self.model.tri_count),
                    getattr(self.model, "tri_materials", None),
                    5,
                    "triangle",
                ),
                (
                    int(self.model.edge_count),
                    getattr(self.model, "edge_bending_properties", None),
                    2,
                    "edge",
                ),
            )
            for count, array, components, name in generic_surface_arrays:
                if not count:
                    continue
                if array is None or tuple(array.shape) != (count, components):
                    raise ValueError(
                        f"Newton generic surface {name} materials do not match the topology."
                    )
                values = np.asarray(array.numpy(), dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Newton generic surface {name} materials must be finite."
                    )
                unsupported_values = values.copy()
                if name == "triangle":
                    # Normal air drag is independent of the optional Style3D elastic material.
                    unsupported_values[:, 3] = 0.0
                if np.any(unsupported_values != 0.0):
                    raise NotImplementedError(
                        "Generic Newton surface elasticity is not supported; "
                        "use zero-valued elastic surface materials for tetrahedral collision "
                        "geometry or build Style3D cloth constraints."
                    )

        has_mass_indices = self._particle_mass_non_diagonal_indices is not None
        has_mass_values = self._particle_mass_non_diagonal_values is not None
        if has_mass_indices != has_mass_values:
            raise ValueError(
                "Particle mass non-diagonal indices and values must be provided together."
            )
        if has_mass_indices and len(self._particle_mass_non_diagonal_indices) != len(
            self._particle_mass_non_diagonal_values
        ):
            raise ValueError(
                "Particle mass non-diagonal indices and values have different lengths."
            )

        if self.model.body_count and self._affine_body_ids is None:
            raise NotImplementedError(
                "Newton rigid bodies require style3d_sim.ModelBuilder so affine particle proxies are available."
            )

        if self._affine_body_ids is not None:
            affine_body_ids = [int(body) for body in self._affine_body_ids.numpy()]
            affine_body_count = len(affine_body_ids)
            if len(set(affine_body_ids)) != affine_body_count or any(
                body < 0 or body >= self.model.body_count for body in affine_body_ids
            ):
                raise ValueError(
                    "Affine body ids must be unique valid Newton body indices."
                )
            if (
                self._affine_body_indices is None
                or len(self._affine_body_indices) != affine_body_count
            ):
                raise ValueError(
                    "Affine body particle-index mapping does not match the affine body count."
                )
            if (
                self._affine_body_masses is None
                or len(self._affine_body_masses) != affine_body_count
            ):
                raise ValueError(
                    "Affine body masses do not match the affine body count."
                )
            if (
                self._affine_body_inertia is None
                or len(self._affine_body_inertia) != affine_body_count
            ):
                raise ValueError(
                    "Affine body inertia tensors do not match the affine body count."
                )
            if (
                self._affine_body_com is None
                or len(self._affine_body_com) != affine_body_count
            ):
                raise ValueError(
                    "Affine body centers of mass do not match the affine body count."
                )

            proxy_flags = int(_newton.BodyFlags.DYNAMIC | _newton.BodyFlags.KINEMATIC)
            expected_body_ids = [
                body
                for body, flags in enumerate(self.model.body_flags.numpy())
                if int(flags) & proxy_flags
            ]
            if affine_body_ids != expected_body_ids:
                raise ValueError(
                    "Every dynamic or kinematic Newton body must have exactly one affine proxy mapping in body order."
                )

        body_spring_arrays = (
            self._body_spring_indices,
            self._body_spring_points0,
            self._body_spring_points1,
            self._body_spring_stiffnesses,
            self._body_spring_rest_lengths,
        )
        if any(array is not None for array in body_spring_arrays):
            if any(array is None for array in body_spring_arrays):
                raise ValueError("All body spring arrays must be provided together.")
            body_spring_count = len(self._body_spring_indices)
            if any(len(array) != body_spring_count for array in body_spring_arrays):
                raise ValueError("Body spring arrays have different lengths.")

            affine_body_count = (
                0 if self._affine_body_ids is None else len(self._affine_body_ids)
            )
            for body0, body1 in self._body_spring_indices.numpy():
                if (
                    int(body0) < -1
                    or int(body1) < -1
                    or (int(body0) >= affine_body_count and int(body0) != -1)
                    or (int(body1) >= affine_body_count and int(body1) != -1)
                    or (int(body0) == -1 and int(body1) == -1)
                ):
                    raise ValueError(
                        "Body spring references an invalid affine body index."
                    )

        body_distance_constraint_arrays = (
            self._body_distance_constraint_indices,
            self._body_distance_constraint_points0,
            self._body_distance_constraint_points1,
            self._body_distance_constraint_weights0,
            self._body_distance_constraint_weights1,
            self._body_distance_constraint_rest_lengths,
            self._body_distance_constraint_mass_scales,
        )
        if any(array is not None for array in body_distance_constraint_arrays):
            if any(array is None for array in body_distance_constraint_arrays):
                raise ValueError(
                    "All body distance constraint arrays must be provided together."
                )
            constraint_count = len(self._body_distance_constraint_indices)
            if any(
                len(array) != constraint_count
                for array in body_distance_constraint_arrays
            ):
                raise ValueError(
                    "Body distance constraint arrays have different lengths."
                )

            affine_body_count = (
                0 if self._affine_body_ids is None else len(self._affine_body_ids)
            )
            for body0, body1 in self._body_distance_constraint_indices.numpy():
                if (
                    int(body0) < -1
                    or int(body1) < -1
                    or (int(body0) >= affine_body_count and int(body0) != -1)
                    or (int(body1) >= affine_body_count and int(body1) != -1)
                    or (int(body0) == -1 and int(body1) == -1)
                ):
                    raise ValueError(
                        "Body distance constraint references an invalid affine body index."
                    )

        body_point_line_arrays = (
            self._body_point_line_indices,
            self._body_point_line_points,
            self._body_point_line_endpoints0,
            self._body_point_line_endpoints1,
            self._body_point_line_stiffnesses,
            self._body_point_line_dampings,
        )
        if any(array is not None for array in body_point_line_arrays):
            if any(array is None for array in body_point_line_arrays):
                raise ValueError(
                    "All body point-line constraint arrays must be provided together."
                )
            body_point_line_count = len(self._body_point_line_indices)
            if any(
                len(array) != body_point_line_count for array in body_point_line_arrays
            ):
                raise ValueError(
                    "Body point-line constraint arrays have different lengths."
                )

            affine_body_count = (
                0 if self._affine_body_ids is None else len(self._affine_body_ids)
            )
            for point_body, line_body in self._body_point_line_indices.numpy():
                if (
                    int(point_body) < 0
                    or int(line_body) < -1
                    or int(point_body) >= affine_body_count
                    or (int(line_body) >= affine_body_count and int(line_body) != -1)
                ):
                    raise ValueError(
                        "Body point-line constraint references an invalid affine body index."
                    )

        joint_drive_arrays = (
            self._joint_drive_types,
            self._joint_drive_body_indices,
            self._joint_drive_parent_origins,
            self._joint_drive_child_origins,
            self._joint_drive_parent_axis_points,
            self._joint_drive_parent_reference_points,
            self._joint_drive_child_reference_points,
            self._joint_drive_dof_indices,
            self._joint_drive_target_indices,
        )
        if any(array is not None for array in joint_drive_arrays):
            if any(array is None for array in joint_drive_arrays):
                raise ValueError("All joint drive arrays must be provided together.")
            drive_count = len(self._joint_drive_types)
            if any(len(array) != drive_count for array in joint_drive_arrays):
                raise ValueError("Joint drive arrays have different lengths.")

        for name in ("joint_armature", "joint_damping", "joint_limit_ke", "joint_limit_kd"):
            values = np.asarray(getattr(self.model, name).numpy())
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain finite non-negative values.")

        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("Joint limits must be ordered and not NaN.")

        # The scalar-coordinate mapping cannot represent ball-joint inertia or
        # actuation. Free-root armature is ignored by Newton's MuJoCo adapter too.
        supported_dofs = set() if self._joint_drive_dof_indices is None else set(self._joint_drive_dof_indices.numpy())
        starts = self.model.joint_qd_start.numpy()
        dimensions = self.model.joint_dof_dim.numpy()
        types = self.model.joint_type.numpy()
        enabled = self.model.joint_enabled.numpy()
        armature = self.model.joint_armature.numpy()
        passive = self.model.joint_damping.numpy()
        modes = self.model.joint_target_mode.numpy()
        stiffness = self.model.joint_target_ke.numpy()
        damping = self.model.joint_target_kd.numpy()
        for joint, (joint_type, start, dims) in enumerate(zip(types, starts, dimensions)):
            if not enabled[joint] or int(joint_type) in (int(_newton.JointType.FREE), int(_newton.JointType.ROD)):
                continue
            if int(joint_type) == int(_newton.JointType.D6) and tuple(dims) == (3, 3):
                continue
            for dof in range(int(start), int(start + sum(dims))):
                if dof in supported_dofs:
                    continue
                active = int(modes[dof]) in (1, 2, 3) and (stiffness[dof] != 0.0 or damping[dof] != 0.0)
                if armature[dof] != 0.0 or passive[dof] != 0.0 or active:
                    raise NotImplementedError(
                        f"Joint {joint} has armature, damping, or actuation on an unsupported non-scalar coordinate. "
                        "Use revolute, prismatic, or supported scalar D6 joints."
                    )

    def _as_span(
        self,
        array: Any | None,
        expected_size: int,
        *,
        storage_components: int = 1,
        expected_dtype: Any | None = None,
    ) -> Span:
        """Create a typed native span over a Newton allocation."""
        if expected_size < 0:
            raise ValueError("Expected array size cannot be negative.")
        if array is None:
            if expected_size:
                raise ValueError("Newton state is missing a required particle array.")
            return Span()

        array_device = getattr(array, "device", None)
        if array_device != self.device:
            raise ValueError(
                f"Expected an array on {self.device}, but received {array_device}."
            )
        if not getattr(array, "is_contiguous", False):
            raise ValueError("Native solver arrays must be contiguous.")
        if expected_dtype is not None and array.dtype != expected_dtype:
            raise ValueError(
                f"Expected a Newton array with dtype {expected_dtype}, "
                f"but received {array.dtype}."
            )

        expected_storage_size = expected_size * storage_components
        if array.size < expected_storage_size:
            raise ValueError(
                f"Expected a Newton array with at least {expected_storage_size} scalar elements, "
                f"but received {array.size}."
            )

        return Span(ptr=array.ptr, size=expected_size)

    def _extract_triangle_drag_coefficients(self) -> Any | None:
        """Keep Newton's complete triangle order, including zero-drag tetrahedral surfaces."""
        count = int(self.model.tri_count)
        if count == 0:
            return None
        materials = getattr(self.model, "tri_materials", None)
        if materials is None or tuple(materials.shape) != (count, 5):
            raise ValueError("Newton triangle materials must have shape (tri_count, 5).")
        coefficients = np.asarray(materials.numpy()[:, 3], dtype=np.float32)
        if not np.isfinite(coefficients).all() or np.any(coefficients < 0.0):
            raise ValueError("Newton triangle drag coefficients must be finite and non-negative.")
        if not np.any(coefficients):
            return None
        # The native solver compacts Cd>0 faces once; retaining all coefficients here keeps shared-world validation exact.
        return self._wp.array(coefficients.copy(), dtype=self._wp.float32, device=self.device)

    def _extract_particle_only_shapes(self) -> Any | None:
        """A static infinite plane's signed distance reaches its minimum at a soft feature vertex."""
        if not self.model.shape_count:
            return None
        shape_types = self.model.shape_type.numpy()
        scales = self.model.shape_scale.numpy()
        infinite_planes = (
            (shape_types == int(_newton.GeoType.PLANE))
            & (self.model.shape_body.numpy() < 0)
            & ((scales[:, 0] <= 0) | (scales[:, 1] <= 0))
        )
        if not np.any(infinite_planes):
            return None
        return self._wp.array(infinite_planes.astype(np.int32), dtype=self._wp.int32, device=self.device)

    def _initialize_native_solver(self) -> None:
        """Rebuild native model views and initialize the wrapped solver."""
        state_particle_count = int(self.model.particle_count)
        particle_count = self._shared_counts.get("particle_count", state_particle_count)
        self._synreal_model = Model()
        self._synreal_model.particle_count = particle_count
        self._synreal_model.particle_state_count = state_particle_count
        self._synreal_model.world_count = self._shared_world_count
        self._synreal_model.gravity = self._as_span(
            self.model.gravity,
            self._shared_world_count,
            expected_dtype=self._wp.vec3,
        )
        self._synreal_model.particle_rest_positions = self._as_span(
            self.model.particle_q,
            state_particle_count,
            expected_dtype=self._wp.vec3,
        )

        if particle_count:
            # Synreal uses an explicit particle indirection array. Retain the
            # Warp allocation because the native model stores only its pointer.
            self._particle_indices = self._wp.array(
                list(range(particle_count)),
                dtype=self._wp.int32,
                device=self.model.device,
            )
            self._synreal_model.particle_indices = self._as_span(
                self._particle_indices,
                particle_count,
                expected_dtype=self._wp.int32,
            )
            mass_diagonal = (
                self._particle_mass_diagonal
                if self._particle_mass_diagonal is not None
                else self.model.particle_mass
            )
            self._synreal_model.particle_masses = self._as_span(
                mass_diagonal, particle_count, expected_dtype=self._wp.float32
            )
            mass_non_diagonal_count = self._shared_counts.get(
                "particle_mass_non_diagonal_count",
                (
                    0
                    if self._particle_mass_non_diagonal_indices is None
                    else len(self._particle_mass_non_diagonal_indices)
                ),
            )
            self._synreal_model.particle_mass_non_diagonal_count = (
                mass_non_diagonal_count
            )
            if mass_non_diagonal_count:
                self._synreal_model.particle_mass_non_diagonal_indices = self._as_span(
                    self._particle_mass_non_diagonal_indices,
                    mass_non_diagonal_count,
                    expected_dtype=self._wp.vec2i,
                )
                self._synreal_model.particle_mass_non_diagonal_values = self._as_span(
                    self._particle_mass_non_diagonal_values,
                    mass_non_diagonal_count,
                    expected_dtype=self._wp.float32,
                )
            self._synreal_model.particle_flags = self._as_span(
                self._particle_flags,
                particle_count,
                expected_dtype=self._wp.int32,
            )
            self._synreal_model.particle_world = self._as_span(
                self.model.particle_world,
                particle_count,
                expected_dtype=self._wp.int32,
            )
            self._synreal_model.particle_radii = self._as_span(
                self.model.particle_radius,
                particle_count,
                expected_dtype=self._wp.float32,
            )
            collision_particle_count = self._shared_counts.get(
                "collision_particle_count", self._collision_particle_count
            )
            self._synreal_model.collision_particle_count = collision_particle_count
            if collision_particle_count:
                self._synreal_model.collision_particle_indices = self._as_span(
                    self._collision_particle_indices,
                    collision_particle_count,
                    expected_dtype=self._wp.int32,
                )
        else:
            self._particle_indices = None

        fluid_indices = getattr(self.model, "synreal_fluid_particle_indices", None)
        if fluid_indices is not None:
            self._synreal_model.fluid_particle_count = len(fluid_indices)
            self._synreal_model.fluid_particle_indices = self._as_span(
                fluid_indices, len(fluid_indices), expected_dtype=wp.int32
            )
            for name in ("rest_density", "smoothing_radius", "bulk_modulus", "viscosity"):
                setattr(self._synreal_model, f"fluid_{name}", getattr(self.model, f"synreal_fluid_{name}"))
            boundary_positions = self.model.synreal_fluid_boundary_positions
            boundary_volumes = self.model.synreal_fluid_boundary_volumes
            self._fluid_arrays = (fluid_indices, boundary_positions, boundary_volumes)
            self._synreal_model.fluid_boundary_count = len(boundary_positions)
            if len(boundary_positions):
                self._synreal_model.fluid_boundary_positions = self._as_span(
                    boundary_positions, len(boundary_positions), expected_dtype=wp.vec3
                )
                self._synreal_model.fluid_boundary_volumes = self._as_span(
                    boundary_volumes, len(boundary_volumes), expected_dtype=wp.float32
                )

        granular_indices = getattr(self.model, "synreal_granular_particle_indices", None)
        if granular_indices is not None:
            self._synreal_model.granular_particle_count = len(granular_indices)
            self._synreal_model.granular_particle_indices = self._as_span(
                granular_indices, len(granular_indices), expected_dtype=wp.int32
            )
            for name in (
                "contact_stiffness", "friction", "friction_damping", "cohesion", "bond_damping",
                "yield_strain", "max_compaction", "break_strain",
                "adhesion", "bond_formation_gap", "bond_formation_speed", "max_dynamic_bonds",
            ):
                setattr(self._synreal_model, f"granular_{name}", getattr(self.model, f"synreal_granular_{name}"))
            bonds = self.model.synreal_granular_bond_indices
            lengths = self.model.synreal_granular_bond_rest_lengths
            self._granular_arrays = (granular_indices, bonds, lengths)
            self._synreal_model.granular_bond_count = len(bonds)
            if len(bonds):
                self._synreal_model.granular_bond_indices = self._as_span(bonds, len(bonds), expected_dtype=wp.vec2i)
                self._synreal_model.granular_bond_rest_lengths = self._as_span(
                    lengths, len(lengths), expected_dtype=wp.float32
                )

        spring_count = self._shared_counts.get(
            "spring_count", int(self.model.spring_count)
        )
        self._synreal_model.spring_count = spring_count
        if spring_count:
            self._synreal_model.spring_indices = self._as_span(
                self.model.spring_indices,
                spring_count,
                storage_components=2,
                expected_dtype=self._wp.int32,
            )
            self._synreal_model.spring_stiffnesses = self._as_span(
                self.model.spring_stiffness,
                spring_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.spring_rest_lengths = self._as_span(
                self.model.spring_rest_length,
                spring_count,
                expected_dtype=self._wp.float32,
            )

        tetrahedron_count = self._shared_counts.get(
            "tetrahedron_count", int(self.model.tet_count)
        )
        self._synreal_model.tetrahedron_count = tetrahedron_count
        if tetrahedron_count:
            self._synreal_model.tetrahedron_indices = self._as_span(
                self._tetrahedron_indices,
                tetrahedron_count,
                storage_components=4,
                expected_dtype=self._wp.int32,
            )
            self._synreal_model.tetrahedron_inverse_dms = self._as_span(
                self._tetrahedron_inverse_dms,
                tetrahedron_count,
                expected_dtype=self._wp.mat33,
            )
            self._synreal_model.tetrahedron_materials = self._as_span(
                self._tetrahedron_materials,
                tetrahedron_count,
                storage_components=3,
                expected_dtype=self._wp.float32,
            )

        state_affine_body_count = (
            0 if self._affine_body_ids is None else len(self._affine_body_ids)
        )
        affine_body_count = self._shared_counts.get(
            "body_count", state_affine_body_count
        )
        self._synreal_model.body_count = affine_body_count
        self._synreal_model.body_state_count = state_affine_body_count
        self._synreal_model.cable_count = self._cable_count
        if self._cable_count:
            for field, array, dtype in (
                ("body_indices", self._cable_body_indices, wp.vec2i),
                ("parent_transforms", self._cable_parent_transforms, wp.transform),
                ("child_transforms", self._cable_child_transforms, wp.transform),
                ("stiffnesses", self._cable_stiffnesses, wp.vec4),
                ("dampings", self._cable_dampings, wp.vec4),
                ("rest_bend_twist", self._cable_rest, wp.vec3),
                ("angular_bias", self._cable_bias, wp.vec3),
                ("angular_hessian", self._cable_hessian, wp.vec3),
                ("enabled", self._cable_enabled, wp.int32),
            ):
                setattr(self._synreal_model, f"cable_{field}", self._as_span(array, self._cable_count, expected_dtype=dtype))
        if affine_body_count:
            self._synreal_model.body_masses = self._as_span(
                self._affine_body_masses, affine_body_count, expected_dtype=self._wp.float32
            )
            self._synreal_model.body_inertia_tensors = self._as_span(
                self._affine_body_inertia, affine_body_count, expected_dtype=self._wp.mat33
            )
            self._synreal_model.affine_body_indices = self._as_span(
                self._affine_body_indices, affine_body_count, expected_dtype=self._wp.vec4i
            )
            self._synreal_model.affine_body_ids = self._as_span(
                self._affine_body_ids, affine_body_count, expected_dtype=self._wp.int32
            )
            self._synreal_model.body_centers_of_mass = self._as_span(
                self._affine_body_com, affine_body_count, expected_dtype=self._wp.vec3
            )

        body_spring_count = self._shared_counts.get(
            "body_spring_count",
            (
                0
                if self._body_spring_indices is None
                else len(self._body_spring_indices)
            ),
        )
        self._synreal_model.body_spring_count = body_spring_count
        self._synreal_model.joint_body_spring_count = self._shared_counts.get(
            "joint_body_spring_count", self._joint_body_spring_count
        )
        if body_spring_count:
            self._synreal_model.body_spring_indices = self._as_span(
                self._body_spring_indices, body_spring_count, expected_dtype=self._wp.vec2i
            )
            self._synreal_model.body_spring_points0 = self._as_span(
                self._body_spring_points0, body_spring_count, expected_dtype=self._wp.vec3
            )
            self._synreal_model.body_spring_points1 = self._as_span(
                self._body_spring_points1, body_spring_count, expected_dtype=self._wp.vec3
            )
            self._synreal_model.body_spring_stiffnesses = self._as_span(
                self._body_spring_stiffnesses, body_spring_count, expected_dtype=self._wp.float32
            )
            self._synreal_model.body_spring_rest_lengths = self._as_span(
                self._body_spring_rest_lengths, body_spring_count, expected_dtype=self._wp.float32
            )

        body_distance_constraint_count = self._shared_counts.get(
            "body_distance_constraint_count",
            (
                0
                if self._body_distance_constraint_indices is None
                else len(self._body_distance_constraint_indices)
            ),
        )
        self._synreal_model.body_distance_constraint_count = (
            body_distance_constraint_count
        )
        if body_distance_constraint_count:
            self._synreal_model.body_distance_constraint_indices = self._as_span(
                self._body_distance_constraint_indices,
                body_distance_constraint_count,
                expected_dtype=self._wp.vec2i,
            )
            self._synreal_model.body_distance_constraint_points0 = self._as_span(
                self._body_distance_constraint_points0,
                body_distance_constraint_count,
                expected_dtype=self._wp.vec3,
            )
            self._synreal_model.body_distance_constraint_points1 = self._as_span(
                self._body_distance_constraint_points1,
                body_distance_constraint_count,
                expected_dtype=self._wp.vec3,
            )
            self._synreal_model.body_distance_constraint_weights0 = self._as_span(
                self._body_distance_constraint_weights0,
                body_distance_constraint_count,
                expected_dtype=self._wp.vec4,
            )
            self._synreal_model.body_distance_constraint_weights1 = self._as_span(
                self._body_distance_constraint_weights1,
                body_distance_constraint_count,
                expected_dtype=self._wp.vec4,
            )
            self._synreal_model.body_distance_constraint_mass_scales = self._as_span(
                self._body_distance_constraint_mass_scales,
                body_distance_constraint_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.body_distance_constraint_rest_lengths = self._as_span(
                self._body_distance_constraint_rest_lengths,
                body_distance_constraint_count,
                expected_dtype=self._wp.float32,
            )

        body_point_line_count = self._shared_counts.get(
            "body_point_line_count",
            (
                0
                if self._body_point_line_indices is None
                else len(self._body_point_line_indices)
            ),
        )
        self._synreal_model.body_point_line_count = body_point_line_count
        self._synreal_model.joint_body_point_line_count = self._shared_counts.get(
            "joint_body_point_line_count", self._joint_body_point_line_count
        )
        self._synreal_model.joint_constraint_mass_scale = (
            self._joint_constraint_mass_scale
        )
        if body_point_line_count:
            self._synreal_model.body_point_line_indices = self._as_span(
                self._body_point_line_indices,
                body_point_line_count,
                expected_dtype=self._wp.vec2i,
            )
            self._synreal_model.body_point_line_points = self._as_span(
                self._body_point_line_points,
                body_point_line_count,
                expected_dtype=self._wp.vec3,
            )
            self._synreal_model.body_point_line_endpoints0 = self._as_span(
                self._body_point_line_endpoints0,
                body_point_line_count,
                expected_dtype=self._wp.vec3,
            )
            self._synreal_model.body_point_line_endpoints1 = self._as_span(
                self._body_point_line_endpoints1,
                body_point_line_count,
                expected_dtype=self._wp.vec3,
            )
            self._synreal_model.body_point_line_stiffnesses = self._as_span(
                self._body_point_line_stiffnesses,
                body_point_line_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.body_point_line_dampings = self._as_span(
                self._body_point_line_dampings,
                body_point_line_count,
                expected_dtype=self._wp.float32,
            )

        joint_drive_count = self._shared_counts.get(
            "joint_drive_count",
            (0 if self._joint_drive_types is None else len(self._joint_drive_types)),
        )
        self._synreal_model.joint_drive_count = joint_drive_count
        self._synreal_model.joint_dof_count = self._shared_counts.get(
            "joint_dof_count", int(self.model.joint_dof_count)
        )
        self._synreal_model.joint_target_count = self._shared_counts.get(
            "joint_target_count", len(self.model.joint_target_q)
        )
        if joint_drive_count:
            self._synreal_model.joint_drive_types = self._as_span(
                self._joint_drive_types, joint_drive_count, expected_dtype=self._wp.int32
            )
            self._synreal_model.joint_drive_body_indices = self._as_span(
                self._joint_drive_body_indices, joint_drive_count, expected_dtype=self._wp.vec2i
            )
            self._synreal_model.joint_drive_parent_origins = self._as_span(
                self._joint_drive_parent_origins, joint_drive_count, expected_dtype=self._wp.vec3
            )
            self._synreal_model.joint_drive_child_origins = self._as_span(
                self._joint_drive_child_origins, joint_drive_count, expected_dtype=self._wp.vec3
            )
            self._synreal_model.joint_drive_parent_axis_points = self._as_span(
                self._joint_drive_parent_axis_points, joint_drive_count, expected_dtype=self._wp.vec3
            )
            self._synreal_model.joint_drive_parent_reference_points = self._as_span(
                self._joint_drive_parent_reference_points, joint_drive_count, expected_dtype=self._wp.vec3
            )
            self._synreal_model.joint_drive_child_reference_points = self._as_span(
                self._joint_drive_child_reference_points, joint_drive_count, expected_dtype=self._wp.vec3
            )
            self._synreal_model.joint_drive_dof_indices = self._as_span(
                self._joint_drive_dof_indices, joint_drive_count, expected_dtype=self._wp.int32
            )
            self._synreal_model.joint_drive_target_indices = self._as_span(
                self._joint_drive_target_indices, joint_drive_count, expected_dtype=self._wp.int32
            )
            self._synreal_model.joint_target_stiffnesses = self._as_span(
                self.model.joint_target_ke, self._synreal_model.joint_dof_count, expected_dtype=self._wp.float32
            )
            self._synreal_model.joint_target_dampings = self._as_span(
                self.model.joint_target_kd, self._synreal_model.joint_dof_count, expected_dtype=self._wp.float32
            )
            self._synreal_model.joint_effort_limits = self._as_span(
                self.model.joint_effort_limit, self._synreal_model.joint_dof_count, expected_dtype=self._wp.float32
            )
            self._synreal_model.joint_target_modes = self._as_span(
                self.model.joint_target_mode, self._synreal_model.joint_dof_count, expected_dtype=self._wp.int32
            )
            self._synreal_model.joint_passive_dampings = self._as_span(
                self.model.joint_damping, self._synreal_model.joint_dof_count, expected_dtype=self._wp.float32
            )
            self._synreal_model.joint_armature = self._as_span(
                self.model.joint_armature, self._synreal_model.joint_dof_count, expected_dtype=self._wp.float32
            )
            for native_name, model_name in (
                ("joint_limit_lower", "joint_limit_lower"),
                ("joint_limit_upper", "joint_limit_upper"),
                ("joint_limit_stiffnesses", "joint_limit_ke"),
                ("joint_limit_dampings", "joint_limit_kd"),
            ):
                setattr(self._synreal_model, native_name, self._as_span(
                    getattr(self.model, model_name), self._synreal_model.joint_dof_count, expected_dtype=self._wp.float32
                ))

        shape_count = self._shared_counts.get(
            "shape_count", int(self.model.shape_count)
        )
        self._synreal_model.shape_count = shape_count
        self._synreal_model.shape_replicated_count = self._shared_counts.get(
            "shape_replicated_count", int(self.model.shape_count)
        )
        if shape_count:
            self._synreal_model.shape_bodies = self._as_span(
                self._shared_shape_arrays.get("shape_bodies", self.model.shape_body),
                shape_count,
                expected_dtype=self._wp.int32,
            )
            self._synreal_model.shape_types = self._as_span(
                self._shared_shape_arrays.get("shape_types", self.model.shape_type),
                shape_count,
                expected_dtype=self._wp.int32,
            )
            self._synreal_model.shape_transforms = self._as_span(
                self._shared_shape_arrays.get("shape_transforms", self.model.shape_transform),
                shape_count,
                expected_dtype=self._wp.transform,
            )
            self._synreal_model.shape_scales = self._as_span(
                self._shared_shape_arrays.get("shape_scales", self.model.shape_scale),
                shape_count,
                expected_dtype=self._wp.vec3,
            )
            self._synreal_model.shape_contact_stiffnesses = self._as_span(
                self._shared_shape_arrays.get(
                    "shape_contact_stiffnesses", self.model.shape_material_ke
                ),
                shape_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.shape_contact_dampings = self._as_span(
                self._shared_shape_arrays.get(
                    "shape_contact_dampings", self.model.shape_material_kd
                ),
                shape_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.shape_contact_friction_gains = self._as_span(
                self._shared_shape_arrays.get("shape_contact_friction_gains", self.model.shape_material_kf),
                shape_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.shape_frictions = self._as_span(
                self._shared_shape_arrays.get(
                    "shape_frictions", self.model.shape_material_mu
                ),
                shape_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.shape_rolling_frictions = self._as_span(
                self._shared_shape_arrays.get("shape_rolling_frictions", self.model.shape_material_mu_rolling),
                shape_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.shape_torsional_frictions = self._as_span(
                self._shared_shape_arrays.get("shape_torsional_frictions", self.model.shape_material_mu_torsional),
                shape_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.shape_margins = self._as_span(
                self._shared_shape_arrays.get("shape_margins", self.model.shape_margin),
                shape_count,
                expected_dtype=self._wp.float32,
            )
        self._synreal_model.soft_contact_stiffness = float(self.model.soft_contact_ke)
        self._synreal_model.soft_contact_damping = float(self.model.soft_contact_kd)
        self._synreal_model.soft_contact_friction_gain = float(self.model.soft_contact_kf)
        self._synreal_model.soft_contact_friction = float(self.model.soft_contact_mu)

        triangle_count = self._shared_counts.get(
            "triangle_count", self._native_triangle_count
        )
        self._synreal_model.triangle_count = triangle_count
        if triangle_count:
            self._synreal_model.triangle_indices = self._as_span(
                self.model.tri_indices,
                triangle_count,
                storage_components=3,
                expected_dtype=self._wp.int32,
            )
            self._synreal_model.triangle_rest_areas = self._as_span(
                self.model.tri_areas,
                triangle_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.triangle_inv_dms = self._as_span(
                self.model.tri_poses,
                triangle_count,
                expected_dtype=self._wp.mat22,
            )
            self._synreal_model.triangle_aniso_stiffnesses = self._as_span(
                self._style3d.tri_aniso_ke,
                triangle_count,
                expected_dtype=self._wp.vec3,
            )

        edge_count = self._shared_counts.get("edge_count", self._native_edge_count)
        self._synreal_model.edge_count = edge_count
        if edge_count:
            self._synreal_model.edge_indices = self._as_span(
                self.model.edge_indices,
                edge_count,
                storage_components=4,
                expected_dtype=self._wp.int32,
            )
            self._synreal_model.edge_bending_properties = self._as_span(
                self.model.edge_bending_properties,
                edge_count,
                storage_components=2,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.edge_rest_areas = self._as_span(
                self._style3d.edge_rest_area,
                edge_count,
                expected_dtype=self._wp.float32,
            )
            self._synreal_model.edge_bending_cots = self._as_span(
                self._style3d.edge_bending_cot,
                edge_count,
                expected_dtype=self._wp.vec4,
            )

        collision_triangle_count = self._shared_counts.get(
            "collision_triangle_count", self._collision_triangle_count
        )
        self._synreal_model.collision_triangle_count = collision_triangle_count
        if collision_triangle_count:
            self._synreal_model.collision_triangle_indices = self._as_span(
                self.model.tri_indices,
                collision_triangle_count,
                storage_components=3,
                expected_dtype=self._wp.int32,
            )
            if self._triangle_drag_coefficients is not None:
                self._synreal_model.triangle_drag_coefficients = self._as_span(
                    self._triangle_drag_coefficients, collision_triangle_count, expected_dtype=self._wp.float32
                )

        collision_edge_count = self._shared_counts.get(
            "collision_edge_count", self._collision_edge_count
        )
        self._synreal_model.collision_edge_count = collision_edge_count
        if collision_edge_count:
            self._synreal_model.collision_edge_indices = self._as_span(
                self._collision_edge_indices,
                collision_edge_count,
                storage_components=4,
                expected_dtype=self._wp.int32,
            )

        # Native initialization copies Style3D coefficients to the host while
        # building the fixed PD matrix, so all preceding Warp work must finish.
        self._wp.synchronize_device(self.device)
        with wp.ScopedDevice(self.device):
            self._native_solver.initialize(self._synreal_model)
            # Examples capture their first substep. Allocate history before
            # capture instead of issuing cudaMalloc inside the graph.
            self._reserve_rigid_contact_capacity(int(getattr(self.model, "rigid_contact_max", 0) or 0))

    def _as_synreal_state(self, state: Any) -> State:
        """Wrap a Newton state without taking ownership of its arrays."""
        particle_count = self.model.particle_count
        if state.particle_count != particle_count:
            raise ValueError(
                f"Newton state has {state.particle_count} particles, "
                f"but the solver model has {particle_count}."
            )

        native_state = State()
        native_state.particle_q = self._as_span(
            state.particle_q, particle_count, expected_dtype=self._wp.vec3
        )
        native_state.particle_qd = self._as_span(
            state.particle_qd, particle_count, expected_dtype=self._wp.vec3
        )
        native_state.particle_f = self._as_span(
            state.particle_f, particle_count, expected_dtype=self._wp.vec3
        )
        body_count = int(self.model.body_count)
        if body_count:
            for name, dtype in (
                ("body_q", self._wp.transform),
                ("body_qd", self._wp.spatial_vector),
            ):
                body_array = getattr(state, name, None)
                if body_array is None:
                    raise ValueError(f"Newton state is missing required {name} array.")
                self._as_span(body_array, body_count, expected_dtype=dtype)
            body_force = getattr(state, "body_f", None)
            if body_force is not None:
                self._as_span(body_force, body_count, expected_dtype=self._wp.spatial_vector)
        return native_state

    def _validate_body_state_arrays(self, state: Any) -> None:
        """Validate body arrays used by Python-side affine prediction kernels."""
        body_count = int(self.model.body_count)
        if not body_count:
            return
        self._as_span(state.body_q, body_count, expected_dtype=self._wp.transform)
        self._as_span(state.body_qd, body_count, expected_dtype=self._wp.spatial_vector)

    def sync_affine_body_state(self, state: Any, *, body_mask: Any | None = None) -> None:
        """Update affine proxy positions and velocities from Newton body state.

        Call this after ``eval_fk()``, an environment reset, or any direct edit
        of ``body_q``/``body_qd``. The operation stays on the active CUDA stream
        and is safe to include in CUDA graph capture. It must not be called
        before every regular step because that would discard the proxy state
        integrated by the native solver.

        ``body_mask`` is an optional int32 CUDA array of shape ``(body_count,)``.
        Only selected bodies and their joint-angle histories are reset. Mask the
        preceding FK call too, so surviving environments retain their solver state.
        """
        affine_body_count = (
            0 if self._affine_body_ids is None else len(self._affine_body_ids)
        )
        if not affine_body_count:
            return
        if state.particle_count != self.model.particle_count:
            raise ValueError(
                f"Newton state has {state.particle_count} particles, "
                f"but the solver model has {self.model.particle_count}."
            )
        if state.body_q is None or state.body_qd is None:
            raise ValueError("Affine body synchronization requires body_q and body_qd.")
        self._validate_body_state_arrays(state)
        self._as_span(state.particle_q, self.model.particle_count, expected_dtype=self._wp.vec3)
        self._as_span(state.particle_qd, self.model.particle_count, expected_dtype=self._wp.vec3)

        stream = self._wp.get_stream(self.device)
        if body_mask is None:
            self._native_solver.reset_joint_drive_state(stream)
            self._joint_drive_feedback_state = None
            self._joint_drive_feedback_previous_state = None
            self._joint_drive_feedback_dt = None
            if self._joint_drive_feedback_angle_initialized is not None:
                self._joint_drive_feedback_angle_initialized.zero_()
        else:
            self._as_span(body_mask, self.model.body_count, expected_dtype=self._wp.int32)
            if tuple(body_mask.shape) != (self.model.body_count,):
                raise ValueError("body_mask must have shape (body_count,)")
            if self._joint_drive_types is not None and len(self._joint_drive_types):
                self._ensure_joint_drive_feedback_angle_state()
                drive_mask = self._joint_drive_reset_mask
                self._wp.launch(
                    _mask_joint_drive_reset, dim=len(self._joint_drive_types),
                    inputs=[body_mask, self._affine_body_ids, self._joint_drive_body_indices,
                            self._joint_drive_feedback_angle_initialized],
                    outputs=[drive_mask], device=self.device, stream=stream,
                )
                self._native_solver.reset_joint_drive_state(
                    stream, self._as_span(drive_mask, len(drive_mask), expected_dtype=self._wp.int32)
                )
        self._wp.launch(
            _sync_affine_body_state,
            dim=affine_body_count,
            inputs=[
                self.model.particle_q,
                self._affine_body_indices,
                self._affine_body_ids,
                self._affine_body_com,
                state.body_q,
                state.body_qd,
                body_mask,
            ],
            outputs=[state.particle_q, state.particle_qd],
            device=self.device,
            stream=stream,
        )

    def reset(self, state: Any, world_mask: Any | None = None, flags: Any | None = None) -> None:
        """Restore selected state fields and rebaseline cable hysteresis.

        ``flags=None`` copies body and particle poses/velocities from model defaults;
        ``flags=0`` preserves public state. Cable stress and increments are cleared
        for selected worlds regardless of flags. Cable penalties return to their
        configured floors. Curvature is rebaselined from the next step's input
        pose, allowing FK or direct pose edits after this call.
        Canonical world masks and all device work are safe during CUDA capture.
        After further body edits, call ``sync_affine_body_state`` before stepping.
        """
        if state is None:
            raise ValueError("'state' argument is required.")
        model = self.model
        world_mask = self._normalize_reset_world_mask(world_mask)
        flags_value = int(_newton.StateFlags.ALL if flags is None else flags)
        fields = {}
        for name, flag, count, dtype in (
            ("body_q", _newton.StateFlags.BODY_Q, int(model.body_count), wp.transform),
            ("body_qd", _newton.StateFlags.BODY_QD, int(model.body_count), wp.spatial_vector),
            ("particle_q", _newton.StateFlags.PARTICLE_Q, int(model.particle_count), wp.vec3),
            ("particle_qd", _newton.StateFlags.PARTICLE_QD, int(model.particle_count), wp.vec3),
        ):
            array = getattr(state, name, None) if flags_value & int(flag) else None
            if array is not None:
                self._as_span(array, count, expected_dtype=dtype)
            fields[name] = array
        reset_dim = max(
            int(model.body_count) if fields["body_q"] is not None or fields["body_qd"] is not None else 0,
            int(model.particle_count) if fields["particle_q"] is not None or fields["particle_qd"] is not None else 0,
        )
        if reset_dim:
            wp.launch(
                _reset_newton_state,
                dim=reset_dim,
                inputs=[world_mask, world_mask is None, int(model.world_count), model.body_world, model.particle_world,
                        model.body_q, model.body_qd, model.particle_q, model.particle_qd],
                outputs=[fields["body_q"], fields["body_qd"], fields["particle_q"], fields["particle_qd"]],
                device=self.device,
            )
        # A body reset also updates its derived ABD representation. Restrict the
        # output fields and worlds so unrelated particle state stays untouched.
        proxy_q = getattr(state, "particle_q", None) if fields["body_q"] is not None else None
        proxy_qd = getattr(state, "particle_qd", None) if fields["body_qd"] is not None else None
        body_q = getattr(state, "body_q", None)
        # A velocity-only reset may omit body poses, leaving no axes for proxy velocities.
        if (
            self._affine_body_ids is not None
            and len(self._affine_body_ids)
            and body_q is not None
            and (proxy_q is not None or proxy_qd is not None)
        ):
            self._as_span(body_q, int(model.body_count), expected_dtype=wp.transform)
            if proxy_q is not None:
                self._as_span(proxy_q, int(model.particle_count), expected_dtype=wp.vec3)
            if proxy_qd is not None:
                self._as_span(proxy_qd, int(model.particle_count), expected_dtype=wp.vec3)
            wp.launch(
                _sync_selected_affine_body_state,
                dim=len(self._affine_body_ids),
                inputs=[self._affine_body_indices, self._affine_body_ids, self._affine_body_com,
                        body_q, fields["body_qd"] if proxy_qd is not None else None, model.body_world,
                        world_mask, int(model.world_count), world_mask is None],
                outputs=[proxy_q, proxy_qd],
                device=self.device,
            )
        if self._cable_count:
            wp.launch(
                _reset_cable_history,
                dim=max(self._cable_count, int(model.joint_count), int(model.world_count) + 1),
                inputs=[world_mask, world_mask is None, int(model.world_count), self._cable_joints, model.joint_world],
                outputs=[self._cable_rebaseline_mask, self.joint_sigma_prev, self.joint_kappa_prev,
                         self._joint_dkappa_prev, self._joint_sigma_start, self._joint_C_fric,
                         self._cable_penalty_reset_mask],
                device=self.device,
            )
            # Cable models retain unique rows per actual Newton world. The mask
            # is produced on-device so selected resets remain graph-capturable.
            self._native_solver.reset_cable_penalty_state(
                self._wp.get_stream(self.device),
                self._as_span(self._cable_penalty_reset_mask, self._cable_count, expected_dtype=wp.int32),
            )

    def coupling_notify_input_state_update(
        self, state: Any, flags: Any, *, iteration_restart: bool = False, dt: float = 0.0
    ) -> None:
        """Refresh local affine proxies after the coupler edits rigid poses or twists."""
        del iteration_restart, dt
        if flags & (_newton.StateFlags.BODY_Q | _newton.StateFlags.BODY_QD):
            self.sync_affine_body_state(state)

    def coupling_supports_full_surface_soft_contacts(self) -> bool:
        return True

    def coupling_supports_inertial_property_refresh(self) -> bool:
        """Support ADMM mass/inertia refresh with fixed COM, topology, and mobility.

        Replicated native templates share a mass metric, so heterogeneous
        proximal updates require one native row set per world.
        """
        return self._shared_world_count == 1

    def set_rigid_history_update(self, enabled: bool) -> None:
        """Control history remapping for the next substep after collision detection."""
        self._rigid_history_update = bool(enabled)

    def _as_synreal_contact(self, contacts: Any | None, contact_force: Any | None = None) -> Contact:
        """Wrap Newton rigid- and soft-contact buffers without taking ownership."""
        native_contact = Contact()
        native_contact.history_refresh = getattr(self, "_rigid_history_update", True)
        self._rigid_history_update = True
        if contacts is None:
            if contact_force is not None:
                raise ValueError("Newton contact-force observables require Contacts.")
            return native_contact

        rigid_capacity = int(contacts.rigid_contact_max)
        if rigid_capacity < 0:
            raise ValueError("Newton rigid contact capacity cannot be negative.")
        soft_capacity = int(contacts.soft_contact_max)
        if soft_capacity < 0:
            raise ValueError("Newton soft contact capacity cannot be negative.")
        if contact_force is not None:
            native_contact.contact_force = self._as_span(
                contact_force,
                rigid_capacity + soft_capacity,
                expected_dtype=self._wp.spatial_vector,
            )
        if rigid_capacity:
            if self.config.rigid_contact_hard:
                self._reserve_rigid_contact_capacity(rigid_capacity)
            native_contact.max_count = rigid_capacity
            match_index = getattr(contacts, "rigid_contact_match_index", None)
            if match_index is not None:
                native_contact.match_index = self._as_span(match_index, rigid_capacity, expected_dtype=self._wp.int32)
            native_contact.count = self._as_span(
                contacts.rigid_contact_count, 1, expected_dtype=self._wp.int32
            )
            native_contact.shape0 = self._as_span(
                contacts.rigid_contact_shape0,
                rigid_capacity,
                expected_dtype=self._wp.int32,
            )
            native_contact.shape1 = self._as_span(
                contacts.rigid_contact_shape1,
                rigid_capacity,
                expected_dtype=self._wp.int32,
            )
            native_contact.point0 = self._as_span(
                contacts.rigid_contact_point0,
                rigid_capacity,
                expected_dtype=self._wp.vec3,
            )
            native_contact.point1 = self._as_span(
                contacts.rigid_contact_point1,
                rigid_capacity,
                expected_dtype=self._wp.vec3,
            )
            # Capsule records store centerline points separately from radial
            # surface offsets. Friction needs the complete surface lever arm.
            for side in (0, 1):
                offset = getattr(contacts, f"rigid_contact_offset{side}", None)
                if offset is not None:
                    setattr(native_contact, f"offset{side}", self._as_span(
                        offset, rigid_capacity, expected_dtype=self._wp.vec3
                    ))
            native_contact.normal = self._as_span(
                contacts.rigid_contact_normal,
                rigid_capacity,
                expected_dtype=self._wp.vec3,
            )
            native_contact.margin0 = self._as_span(
                contacts.rigid_contact_margin0,
                rigid_capacity,
                expected_dtype=self._wp.float32,
            )
            native_contact.margin1 = self._as_span(
                contacts.rigid_contact_margin1,
                rigid_capacity,
                expected_dtype=self._wp.float32,
            )

        if soft_capacity:
            native_contact.soft_max_count = soft_capacity
            native_contact.soft_count = self._as_span(
                contacts.soft_contact_count, 1, expected_dtype=self._wp.int32
            )
            # Full-surface records can identify the deformable side with three
            # vertices and barycentric weights, so the legacy particle id is
            # optional unless particle-only shape fallback is enabled.
            soft_particle = getattr(contacts, "soft_contact_particle", None)
            if soft_particle is not None:
                native_contact.soft_particle = self._as_span(
                    soft_particle,
                    soft_capacity,
                    expected_dtype=self._wp.int32,
                )
            if getattr(contacts, "_enable_rigid_soft_full_surface_contact", False):
                native_contact.soft_indices = self._as_span(
                    contacts.soft_contact_indices, soft_capacity, expected_dtype=self._wp.vec3i
                )
                native_contact.soft_barycentric = self._as_span(
                    contacts.soft_contact_barycentric, soft_capacity, expected_dtype=self._wp.vec3
                )
                if self._soft_particle_only_shapes is not None:
                    native_contact.soft_particle_only_shapes = self._as_span(
                        self._soft_particle_only_shapes, int(self.model.shape_count), expected_dtype=self._wp.int32
                    )
            native_contact.soft_shape = self._as_span(
                contacts.soft_contact_shape,
                soft_capacity,
                expected_dtype=self._wp.int32,
            )
            native_contact.soft_body_pos = self._as_span(
                contacts.soft_contact_body_pos,
                soft_capacity,
                expected_dtype=self._wp.vec3,
            )
            native_contact.soft_body_vel = self._as_span(
                contacts.soft_contact_body_vel,
                soft_capacity,
                expected_dtype=self._wp.vec3,
            )
            native_contact.soft_normal = self._as_span(
                contacts.soft_contact_normal,
                soft_capacity,
                expected_dtype=self._wp.vec3,
            )
        return native_contact

    def _as_synreal_control(self, control: Any | None) -> Control:
        """Wrap Newton control buffers without taking ownership."""
        native_control = Control()

        tetrahedron_state_count = int(self.model.tet_count)
        if tetrahedron_state_count and control is not None:
            tetrahedron_activations = getattr(control, "tet_activations", None)
            if tetrahedron_activations is not None:
                if (
                    tuple(tetrahedron_activations.shape) != (tetrahedron_state_count,)
                    or tetrahedron_activations.dtype != self._wp.float32
                ):
                    raise ValueError(
                        "Newton control tetrahedron activations must be a float32 "
                        f"array with shape ({tetrahedron_state_count},)."
                    )
                native_control.tetrahedron_activations = self._as_span(
                    tetrahedron_activations,
                    tetrahedron_state_count,
                    expected_dtype=self._wp.float32,
                )

        if not self._synreal_model.joint_drive_count:
            return native_control
        if control is None:
            raise ValueError("A Newton control is required for driven joints.")

        joint_dof_state_count = (
            self._synreal_model.joint_dof_count * self._shared_world_count
        )
        joint_target_state_count = (
            self._synreal_model.joint_target_count * self._shared_world_count
        )
        native_control.joint_f = self._as_span(
            control.joint_f,
            joint_dof_state_count,
            expected_dtype=self._wp.float32,
        )
        native_control.joint_target_q = self._as_span(
            control.joint_target_q,
            joint_target_state_count,
            expected_dtype=self._wp.float32,
        )
        native_control.joint_target_qd = self._as_span(
            control.joint_target_qd,
            joint_dof_state_count,
            expected_dtype=self._wp.float32,
        )
        native_control.joint_drive_velocities = self._as_span(
            self._joint_drive_start_velocities,
            len(self._joint_drive_types),
            expected_dtype=self._wp.float32,
        )
        return native_control

    def step(
        self,
        state_in: Any,
        state_out: Any,
        control: Any | None,
        contacts: Any | None,
        dt: float,
        *,
        observables: _newton.solvers.SolverBase.Observables | None = None,
    ) -> None:
        """Advance Newton state arrays through the native Style3D Sim solver.

        Args:
            state_in: Newton input state.
            state_out: Newton output state.
            control: Newton generalized joint-control input.
            contacts: Newton collision-pipeline contact data.
            dt: Time-step duration in seconds.
            observables: Optional Newton diagnostic arrays allocated by this solver.

        When affine-body inertial prediction is enabled, Python integrates the
        rigid-body inertia state including the applied body wrench, then converts
        its pose displacement into affine-proxy secant velocities. Joint
        actuator forces use MuJoCo's mode and actuator-limit semantics. Position
        error is evaluated before integration; actuator and passive damping are
        linearized implicitly in velocity, as in MuJoCo's ``implicitfast``.
        Gravity, particle forces, joint constraints, and contacts retain their
        native paths.
        The native solver then applies affine rigidity, joints, and contacts to
        Cartesian particle positions and velocities. On return, both particle-
        position buffers contain the new positions, while Python reconstructs
        ``state_out.body_q`` and ``state_out.body_qd`` from the resulting proxy
        position and linear-velocity fields. In ``InertiaMode.NONE``, the
        returned proxy velocities are projected onto the rigid velocity field
        represented by ``state_out.body_qd`` so the two representations round-trip.
        """
        self.validate_observables(observables, contacts)
        contact_force = None
        if observables is not None and observables.is_requested(self.ObservableKind.CONTACT_F):
            contact_force = observables.contact_f
            if contact_force.requires_grad:
                raise NotImplementedError("NewtonSolver contact_f observables do not support gradients.")

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and greater than zero")

        native_state_in = self._as_synreal_state(state_in)
        native_state_out = self._as_synreal_state(state_out)
        self._validate_body_state_arrays(state_in)
        self._validate_body_state_arrays(state_out)
        native_control = self._as_synreal_control(control)
        native_contact = self._as_synreal_contact(contacts, contact_force)
        stream = self._wp.get_stream(self.device)

        if self.model.body_count:
            self._wp.copy(self._body_qd_previous, state_in.body_qd, stream=stream)
            self._wp.copy(state_out.body_q, state_in.body_q, stream=stream)
            self._wp.copy(state_out.body_qd, state_in.body_qd, stream=stream)

        if not self.model.particle_count:
            if self._joint_drive_effort is not None:
                self._joint_drive_effort.zero_()
            self._write_body_qdd(state_out, observables, dt, stream)
            return

        if self._cable_count:
            self._prepare_cable_dahl(state_in)
        if self._joint_drive_types is not None and len(self._joint_drive_types):
            # The rigid predictor overwrites particle_qd with its secant target.
            # Capture physical joint velocity before that overwrite for both
            # joint-space armature and the implicit velocity linearization.
            self.compute_joint_drive_effort(
                state_in,
                control,
                out=(self._joint_drive_effort if self._joint_drive_effort is not None
                     else self._joint_drive_feedback_scratch),
                _advance_angle_state=True,
            )

        affine_body_state_count = (
            0 if self._affine_body_ids is None else len(self._affine_body_ids)
        )
        inertia_mode = int(self.config.inertia_mode.value)
        use_rigid_body_inertia = bool(
            affine_body_state_count and inertia_mode != NONE
        )
        if use_rigid_body_inertia:
            self._wp.launch(
                _predict_affine_body_inertia,
                dim=affine_body_state_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    self._affine_body_ids,
                    self.model.body_com,
                    self.model.body_mass,
                    self._affine_body_inertia,
                    self._fixed_affine_body_flags,
                    inertia_mode,
                    dt,
                ],
                outputs=[
                    self._body_prediction_state.body_q,
                    self._body_prediction_state.body_qd,
                ],
                device=self.device,
                stream=stream,
            )
            self._wp.launch(
                _predict_affine_body_velocities,
                dim=affine_body_state_count,
                inputs=[
                    state_in.body_q,
                    self._body_prediction_state.body_q,
                    self._affine_body_indices,
                    self._affine_body_ids,
                    self.model.body_com,
                    state_in.particle_q,
                    dt,
                ],
                outputs=[state_in.particle_qd],
                device=self.device,
                stream=stream,
            )
        elif (
            affine_body_state_count
            and state_in.body_f is not None
            and state_in.particle_f is not None
        ):
            # Compatibility path for the legacy free-particle inertia target.
            # The rigid predictor above consumes the wrench directly and must
            # never enter this scatter path as well.
            self._wp.launch(
                _scatter_affine_body_forces,
                dim=affine_body_state_count,
                inputs=[
                    state_in.body_f,
                    self._affine_body_indices,
                    self._affine_body_ids,
                    state_in.particle_q,
                ],
                outputs=[state_in.particle_f],
                device=self.device,
                stream=stream,
            )
        if affine_body_state_count and (self._fixed_affine_body_ids or self._rotation_locked_affine_body_ids):
            self._wp.launch(
                _sync_fixed_affine_body_state,
                dim=affine_body_state_count,
                inputs=[self._affine_body_indices, self._affine_body_ids, self._fixed_affine_body_flags,
                        self._rotation_locked_affine_body_flags,
                        self._affine_body_com, state_in.body_q, dt],
                outputs=[state_in.particle_q, state_in.particle_qd],
                device=self.device,
                stream=stream,
            )
        self._native_solver.step(
            native_state_in,
            native_state_out,
            native_control,
            native_contact,
            dt,
            stream,
        )

        if affine_body_state_count:
            if self._fixed_affine_bodies is not None:
                self._wp.launch(
                    _clear_fixed_affine_body_velocities,
                    dim=len(self._fixed_affine_body_ids),
                    inputs=[
                        state_out.particle_qd,
                        self._affine_body_indices,
                        self._fixed_affine_bodies,
                    ],
                    device=self.device,
                    stream=stream,
                )
            self._wp.launch(
                _update_affine_body_poses,
                dim=affine_body_state_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    self._affine_body_indices,
                    self._affine_body_ids,
                    self._fixed_affine_body_flags,
                    self.model.body_com,
                    self._affine_body_inertia,
                    inertia_mode,
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=self.device,
                stream=stream,
            )

        self._write_body_qdd(state_out, observables, dt, stream)

        if self._cable_count:
            self._update_cable_dahl(state_out)

        if self._joint_drive_types is not None and len(self._joint_drive_types):
            # Advance only the continuous-angle branch after integration; public
            # telemetry retains the pre-integration sample above.
            self._joint_drive_feedback_state = state_out
            self._joint_drive_feedback_previous_state = state_in
            self._joint_drive_feedback_dt = float(dt)
            self.compute_joint_drive_effort(
                state_out,
                control,
                out=self._joint_drive_feedback_scratch,
                _advance_angle_state=True,
            )

    def _write_body_qdd(
        self,
        state_out: Any,
        observables: _newton.solvers.SolverBase.Observables | None,
        dt: float,
        stream: Any,
    ) -> None:
        body_state_count = self._body_qd_previous.shape[0]
        if (
            observables is None
            or not observables.is_requested(self.ObservableKind.BODY_QDD)
            or not body_state_count
        ):
            return
        self._wp.launch(
            _compute_body_qdd,
            dim=body_state_count,
            inputs=[self._body_qd_previous, state_out.body_qd, 1.0 / dt],
            outputs=[observables.body_qdd],
            device=self.device,
            stream=stream,
        )

    def notify_model_changed(self, flags: Any) -> None:
        """Refresh native views after Newton model arrays are modified.

        Eager BODY_INERTIAL_PROPERTIES updates rebuild cached point weights and
        masks when COM or mobility changes. During graph capture, this flag
        supports only finite mass/inertia metric updates with fixed COM and
        mobility, as used by ADMM; other changes require an eager notification
        and recapture.
        """
        capturing = self._wp.get_stream(self.device).is_capturing
        if int(flags) == int(_newton.ModelFlags.BODY_INERTIAL_PROPERTIES) and self._shared_world_count == 1:
            # ADMM changes finite mass/inertia without changing ownership or
            # mobility. Eager COM/mobility edits still require full precompute;
            # no host reads or allocations may enter the captured metric path.
            if capturing or self._affine_inertial_refresh_preserves_layout():
                self._refresh_affine_mass_properties_device()
                return
        if capturing:
            raise RuntimeError("Model changes are not supported during CUDA graph capture.")
        if flags & (_newton.ModelFlags.BODY_INERTIAL_PROPERTIES | _newton.ModelFlags.BODY_PROPERTIES):
            self._refresh_affine_mass_properties()
        if flags & (_newton.ModelFlags.JOINT_PROPERTIES | _newton.ModelFlags.JOINT_DOF_PROPERTIES):
            self._refresh_cable_materials()
        if flags & (_newton.ModelFlags.JOINT_PROPERTIES | _newton.ModelFlags.BODY_PROPERTIES):
            self._refresh_cable_rest_bend_twist()
            if self._cable_count:
                self._gather_cable_parameters()
        self._triangle_drag_coefficients = self._extract_triangle_drag_coefficients()
        self._soft_particle_only_shapes = self._extract_particle_only_shapes()
        self._body_distance_constraint_mass_scales = (
            self._normalize_body_distance_constraint_mass_scales()
        )
        self._prepare_affine_inertial_refresh()
        self._validate_model()
        self._configure_shared_world_layout()
        self._initialize_native_solver()
        self._joint_drive_feedback_state = None
        self._joint_drive_feedback_previous_state = None
        self._joint_drive_feedback_dt = None
        self._joint_drive_feedback_previous_angles = None
        self._joint_drive_feedback_angle_initialized = None
        self._joint_drive_reset_mask = None
        self._joint_drive_feedback_scratch = None
        self._joint_drive_start_velocities = None
        if self._joint_drive_effort is not None:
            if self._wp.get_stream(self.device).is_capturing:
                raise RuntimeError("Model changes are not supported during CUDA graph capture.")
            self._joint_drive_effort = self._wp.zeros(
                int(self.model.joint_dof_count), dtype=self._wp.float32, device=self.device
            )
        self._ensure_joint_drive_feedback_angle_state()
