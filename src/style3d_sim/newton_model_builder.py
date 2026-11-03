# SPDX-FileCopyrightText: Copyright (c) 2026 Zhejiang Linctex Digital Technology Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Optional Newton ModelBuilder with affine-body particle proxies."""

from __future__ import annotations

from importlib import import_module
from itertools import combinations
from typing import Any

__all__ = ["ModelBuilder"]


def _load_newton() -> Any:
    """Load the optional Newton dependency when this adapter is imported."""
    try:
        return import_module("newton")
    except ModuleNotFoundError as error:
        if error.name == "newton":
            raise ModuleNotFoundError(
                "ModelBuilder requires the optional Newton dependencies. "
                "Install them with `pip install newton`."
            ) from error
        raise


_newton = _load_newton()
np = import_module("numpy")
wp = import_module("warp")


class ModelBuilder(_newton.ModelBuilder):
    """Build Newton models with particle proxies for dynamic rigid bodies.

    Four vector-valued particles store the ABD generalized coordinates of every
    dynamic body: its center of mass followed by the three columns of its affine
    matrix. The native solver can preserve the frame with six proxy-edge springs, the
    original ABD orthogonality potential, or M-ABD-style co-rotated linear
    elasticity. A four-by-four scalar block mass matrix preserves the body's
    translational mass and inertia tensor in affine coordinates.

    The proxy topology is added immediately before finalization, after attached
    shapes have contributed their mass properties. The resulting particles are
    ordinary Newton particles and are therefore visible in Newton viewers.
    """

    def __init__(
        self,
        up_axis: Any = _newton.Axis.Z,
        gravity: float = -9.81,
        *,
        affine_body_particle_radius: float | None = None,
    ) -> None:
        """Create a builder that generates affine-body particle proxies.

        Args:
            up_axis: Newton model up axis.
            gravity: Scalar acceleration along the negative up axis, or an explicit world-space vector.
            affine_body_particle_radius: Viewer and collision radius of proxy
                particles. ``None`` uses Newton's default particle radius.
        """
        if np.isscalar(gravity):
            gravity_vector = np.zeros(3, dtype=np.float32)
            gravity_vector[_newton.Axis.from_any(up_axis).value] = gravity
            gravity = gravity_vector
        super().__init__(up_axis=up_axis, gravity=gravity)
        # SolverNewton treats contacts implicitly, so use a substantially stiffer penalty than Newton's
        # general-purpose explicit-contact default. Keep damping moderate: excessive damping makes resting
        # contact look compliant and removes too much impact velocity before the positional solve converges.
        self.default_shape_cfg.ke = 5.0e6
        self.default_shape_cfg.kd = 1.0e4

        if affine_body_particle_radius is not None and (
            not np.isfinite(affine_body_particle_radius)
            or affine_body_particle_radius < 0.0
        ):
            raise ValueError("Affine-body particle radius cannot be negative.")

        self.affine_body_particle_radius = affine_body_particle_radius
        self.affine_body_ids: list[int] = []
        self.affine_body_particle_indices: list[tuple[int, int, int, int]] = []
        self.affine_body_particle_mass_matrices: list[np.ndarray] = []
        self.body_spring_specs: list[
            tuple[int, int, np.ndarray, np.ndarray, float, float | None]
        ] = []
        self.body_distance_constraint_specs: list[
            tuple[int, int, np.ndarray, np.ndarray, float | None]
        ] = []
        self.body_point_line_specs: list[
            tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float, float]
        ] = []
        self._affine_body_proxies_added = False
        self._synreal_model_finalized = False

    @staticmethod
    def _proxy_mass_matrix(
        body_index: int,
        mass: float,
        inertia: Any,
    ) -> np.ndarray:
        """Build the direct ABD mass block."""
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"Body {body_index} must have positive finite mass.")

        inertia_matrix = np.asarray(inertia, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(inertia_matrix).all():
            raise ValueError(f"Body {body_index} has a non-finite inertia tensor.")
        magnitude = max(
            float(np.max(np.abs(inertia_matrix))), np.finfo(np.float64).tiny
        )
        tolerance = 32.0 * np.finfo(np.float32).eps * magnitude
        if float(np.max(np.abs(inertia_matrix - inertia_matrix.T))) > tolerance:
            raise ValueError(f"Body {body_index} has a non-symmetric inertia tensor.")
        inertia_matrix = 0.5 * (inertia_matrix + inertia_matrix.T)

        # For x = p + A*q, the ABD rotational mass term uses the second moment
        # D = 0.5*trace(J)*I - J. Direct coordinates q=(p, A[:,0], A[:,1], A[:,2]) therefore have mass block
        # diag(m, D); every scalar entry multiplies a world-space 3x3 identity.
        second_moment = (
            0.5 * np.trace(inertia_matrix) * np.eye(3, dtype=np.float64)
            - inertia_matrix
        )
        if float(np.min(np.linalg.eigvalsh(second_moment))) <= tolerance:
            raise ValueError(
                f"Body {body_index} has an invalid inertia tensor with a singular second moment."
            )
        generalized_mass = np.zeros((4, 4), dtype=np.float64)
        generalized_mass[0, 0] = mass
        generalized_mass[1:, 1:] = second_moment

        if not np.isfinite(generalized_mass).all() or not np.allclose(
            generalized_mass, generalized_mass.T, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                f"Body {body_index} produced an invalid affine generalized mass matrix."
            )

        return generalized_mass.astype(np.float32)

    def add_body_spring(
        self,
        body0: int,
        body1: int,
        point0: Any = (0.0, 0.0, 0.0),
        point1: Any = (0.0, 0.0, 0.0),
        *,
        stiffness: float = 1.0e5,
        rest_length: float | None = None,
    ) -> int:
        """Add a spring between rigid-body points or a world-space anchor.

        Args:
            body0: Newton body index containing the first point, or ``-1`` for
                a fixed world-space point.
            body1: Newton body index containing the second point, or ``-1`` for
                a fixed world-space point.
            point0: First point in ``body0`` local coordinates, or world
                coordinates when ``body0`` is ``-1``.
            point1: Second point in ``body1`` local coordinates, or world
                coordinates when ``body1`` is ``-1``.
            stiffness: Spring stiffness in newtons per meter.
            rest_length: Unstrained length in meters. ``None`` uses the distance
                between the two points in the finalized initial pose.

        Returns:
            The index of the appended body spring.
        """
        body_count = len(self.body_q)
        if body0 < -1 or body0 >= body_count or body1 < -1 or body1 >= body_count:
            raise ValueError("Body spring references an invalid Newton body index.")
        if body0 == -1 and body1 == -1:
            raise ValueError("Body spring requires at least one dynamic body endpoint.")
        if body0 == body1:
            raise ValueError("Body spring endpoints must belong to different bodies.")

        local_point0 = np.asarray(point0, dtype=np.float64)
        local_point1 = np.asarray(point1, dtype=np.float64)
        if local_point0.shape != (3,) or local_point1.shape != (3,):
            raise ValueError(
                "Body spring points must contain exactly three components."
            )
        if not np.isfinite(local_point0).all() or not np.isfinite(local_point1).all():
            raise ValueError("Body spring points must be finite.")
        if not np.isfinite(stiffness) or stiffness < 0.0:
            raise ValueError("Body spring stiffness must be finite and non-negative.")
        if rest_length is not None and (
            not np.isfinite(rest_length) or rest_length < 0.0
        ):
            raise ValueError("Body spring rest length must be finite and non-negative.")

        self.body_spring_specs.append(
            (
                int(body0),
                int(body1),
                local_point0.copy(),
                local_point1.copy(),
                float(stiffness),
                None if rest_length is None else float(rest_length),
            )
        )
        return len(self.body_spring_specs) - 1

    def _finalize_body_springs(
        self, body_ids: list[int]
    ) -> tuple[
        list[tuple[int, int]],
        list[np.ndarray],
        list[np.ndarray],
        list[float],
        list[float],
    ]:
        """Validate body springs and convert Newton body ids to affine-body ids."""
        affine_body_by_newton_body = {
            body: affine_body for affine_body, body in enumerate(body_ids)
        }
        indices: list[tuple[int, int]] = []
        points0: list[np.ndarray] = []
        points1: list[np.ndarray] = []
        stiffnesses: list[float] = []
        rest_lengths: list[float] = []

        for (
            body0,
            body1,
            point0,
            point1,
            stiffness,
            rest_length,
        ) in self.body_spring_specs:
            if any(
                body >= 0 and body not in affine_body_by_newton_body
                for body in (body0, body1)
            ):
                raise NotImplementedError(
                    "Body spring body endpoints must belong to dynamic affine bodies."
                )
            if body0 >= 0 and body1 >= 0 and int(self.body_world[body0]) != int(
                self.body_world[body1]
            ):
                raise ValueError("Body spring endpoints must belong to the same world.")

            if rest_length is None:
                world_points = []
                for body, point in ((body0, point0), (body1, point1)):
                    if body == -1:
                        world_points.append(point)
                        continue
                    body_transform = np.asarray(self.body_q[body], dtype=np.float64)
                    rotation = np.asarray(
                        wp.quat_to_matrix(wp.quat(*body_transform[3:7])),
                        dtype=np.float64,
                    ).reshape(3, 3)
                    world_points.append(body_transform[:3] + rotation @ point)
                rest_length = float(np.linalg.norm(world_points[1] - world_points[0]))

            indices.append(
                (
                    -1 if body0 == -1 else affine_body_by_newton_body[body0],
                    -1 if body1 == -1 else affine_body_by_newton_body[body1],
                )
            )
            points0.append(point0.astype(np.float32))
            points1.append(point1.astype(np.float32))
            stiffnesses.append(stiffness)
            rest_lengths.append(rest_length)

        return indices, points0, points1, stiffnesses, rest_lengths

    def add_body_distance_constraint(
        self,
        body0: int,
        body1: int,
        point0: Any = (0.0, 0.0, 0.0),
        point1: Any = (0.0, 0.0, 0.0),
        *,
        rest_length: float | None = None,
    ) -> int:
        """Add an augmented-Lagrangian hard distance constraint.

        Args:
            body0: Newton body index containing the first point, or ``-1`` for
                a fixed world-space point.
            body1: Newton body index containing the second point, or ``-1`` for
                a fixed world-space point.
            point0: First point in ``body0`` local coordinates, or world
                coordinates when ``body0`` is ``-1``.
            point1: Second point in ``body1`` local coordinates, or world
                coordinates when ``body1`` is ``-1``.
            rest_length: Constrained length in meters. ``None`` uses the
                distance between the two points in the finalized initial pose.

        Returns:
            The index of the appended body distance constraint.
        """
        body_count = len(self.body_q)
        if body0 < -1 or body0 >= body_count or body1 < -1 or body1 >= body_count:
            raise ValueError(
                "Body distance constraint references an invalid Newton body index."
            )
        if body0 == -1 and body1 == -1:
            raise ValueError(
                "Body distance constraint requires at least one dynamic body endpoint."
            )
        if body0 == body1:
            raise ValueError(
                "Body distance constraint endpoints must belong to different bodies."
            )

        local_point0 = np.asarray(point0, dtype=np.float64)
        local_point1 = np.asarray(point1, dtype=np.float64)
        if local_point0.shape != (3,) or local_point1.shape != (3,):
            raise ValueError(
                "Body distance constraint points must contain exactly three components."
            )
        if not np.isfinite(local_point0).all() or not np.isfinite(local_point1).all():
            raise ValueError("Body distance constraint points must be finite.")
        if rest_length is not None and (
            not np.isfinite(rest_length) or rest_length < 0.0
        ):
            raise ValueError(
                "Body distance constraint rest length must be finite and non-negative."
            )

        self.body_distance_constraint_specs.append(
            (
                int(body0),
                int(body1),
                local_point0.copy(),
                local_point1.copy(),
                None if rest_length is None else float(rest_length),
            )
        )
        return len(self.body_distance_constraint_specs) - 1

    def _finalize_body_distance_constraints(
        self, body_ids: list[int]
    ) -> tuple[
        list[tuple[int, int]],
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
        list[float],
    ]:
        """Convert hard-distance Newton body ids to affine-body ids."""
        affine_body_by_newton_body = {
            body: affine_body for affine_body, body in enumerate(body_ids)
        }
        indices: list[tuple[int, int]] = []
        points0: list[np.ndarray] = []
        points1: list[np.ndarray] = []
        weights0: list[np.ndarray] = []
        weights1: list[np.ndarray] = []
        rest_lengths: list[float] = []

        def point_weights(body: int, point: np.ndarray) -> np.ndarray:
            if body == -1:
                return np.zeros(4, dtype=np.float32)
            offset = point - np.asarray(self.body_com[body])
            return np.asarray(
                (1.0, offset[0], offset[1], offset[2]),
                dtype=np.float32,
            )

        for body0, body1, point0, point1, rest_length in (
            self.body_distance_constraint_specs
        ):
            if any(
                body >= 0 and body not in affine_body_by_newton_body
                for body in (body0, body1)
            ):
                raise NotImplementedError(
                    "Body distance constraint endpoints must belong to dynamic affine bodies."
                )
            if body0 >= 0 and body1 >= 0 and int(self.body_world[body0]) != int(
                self.body_world[body1]
            ):
                raise ValueError(
                    "Body distance constraint endpoints must belong to the same world."
                )

            if rest_length is None:
                world_points = []
                for body, point in ((body0, point0), (body1, point1)):
                    if body == -1:
                        world_points.append(point)
                        continue
                    body_transform = np.asarray(self.body_q[body], dtype=np.float64)
                    rotation = np.asarray(
                        wp.quat_to_matrix(wp.quat(*body_transform[3:7])),
                        dtype=np.float64,
                    ).reshape(3, 3)
                    world_points.append(body_transform[:3] + rotation @ point)
                rest_length = float(np.linalg.norm(world_points[1] - world_points[0]))

            indices.append(
                (
                    -1 if body0 == -1 else affine_body_by_newton_body[body0],
                    -1 if body1 == -1 else affine_body_by_newton_body[body1],
                )
            )
            points0.append(point0.astype(np.float32))
            points1.append(point1.astype(np.float32))
            weights0.append(point_weights(body0, point0))
            weights1.append(point_weights(body1, point1))
            rest_lengths.append(rest_length)

        return indices, points0, points1, weights0, weights1, rest_lengths

    def add_body_point_line(
        self,
        point_body: int,
        line_body: int,
        point: Any = (0.0, 0.0, 0.0),
        line_endpoint0: Any = (-0.5, 0.0, 0.0),
        line_endpoint1: Any = (0.5, 0.0, 0.0),
        *,
        stiffness: float = 1.0e5,
        damping: float = 0.0,
    ) -> int:
        """Constrain a body point to a body-local or world-space segment.

        The force uses the point's projection onto the segment. A projection
        outside the segment is clamped to its nearest endpoint, where the
        constraint becomes a zero-rest-length point spring.

        Args:
            point_body: Newton body index containing the constrained point.
            line_body: Newton body index containing both segment endpoints, or
                ``-1`` when the endpoints are fixed in world space.
            point: Point in ``point_body`` local coordinates.
            line_endpoint0: First endpoint in ``line_body`` local coordinates,
                or world coordinates when ``line_body`` is ``-1``.
            line_endpoint1: Second endpoint in ``line_body`` local coordinates,
                or world coordinates when ``line_body`` is ``-1``.
            stiffness: Constraint stiffness in newtons per meter.
            damping: Constraint-normal damping in newton-seconds per meter.

        Returns:
            The index of the appended body point-line constraint.
        """
        body_count = len(self.body_q)
        if (
            point_body < 0
            or point_body >= body_count
            or line_body < -1
            or line_body >= body_count
        ):
            raise ValueError(
                "Body point-line constraint references an invalid Newton body index."
            )
        if point_body == line_body:
            raise ValueError(
                "Body point-line constraint requires two different bodies."
            )

        local_point = np.asarray(point, dtype=np.float64)
        local_endpoint0 = np.asarray(line_endpoint0, dtype=np.float64)
        local_endpoint1 = np.asarray(line_endpoint1, dtype=np.float64)
        if any(
            value.shape != (3,)
            for value in (local_point, local_endpoint0, local_endpoint1)
        ):
            raise ValueError(
                "Body point-line points must contain exactly three components."
            )
        if not all(
            np.isfinite(value).all()
            for value in (local_point, local_endpoint0, local_endpoint1)
        ):
            raise ValueError("Body point-line points must be finite.")
        if np.array_equal(local_endpoint0, local_endpoint1):
            raise ValueError("Body point-line endpoints must be different.")
        if not np.isfinite(stiffness) or stiffness < 0.0:
            raise ValueError(
                "Body point-line stiffness must be finite and non-negative."
            )
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("Body point-line damping must be finite and non-negative.")

        self.body_point_line_specs.append(
            (
                int(point_body),
                int(line_body),
                local_point.copy(),
                local_endpoint0.copy(),
                local_endpoint1.copy(),
                float(stiffness),
                float(damping),
            )
        )
        return len(self.body_point_line_specs) - 1

    def _finalize_body_point_lines(
        self, body_ids: list[int]
    ) -> tuple[
        list[tuple[int, int]],
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
        list[float],
        list[float],
    ]:
        """Convert Newton body ids in point-line constraints to affine-body ids."""
        affine_body_by_newton_body = {
            body: affine_body for affine_body, body in enumerate(body_ids)
        }
        indices: list[tuple[int, int]] = []
        points: list[np.ndarray] = []
        endpoints0: list[np.ndarray] = []
        endpoints1: list[np.ndarray] = []
        stiffnesses: list[float] = []
        dampings: list[float] = []

        for (
            point_body,
            line_body,
            point,
            endpoint0,
            endpoint1,
            stiffness,
            damping,
        ) in self.body_point_line_specs:
            if point_body not in affine_body_by_newton_body or (
                line_body >= 0 and line_body not in affine_body_by_newton_body
            ):
                raise NotImplementedError(
                    "Body point-line body endpoints must belong to dynamic affine bodies."
                )
            if line_body >= 0 and int(self.body_world[point_body]) != int(
                self.body_world[line_body]
            ):
                raise ValueError(
                    "Body point-line constraint bodies must belong to the same world."
                )

            affine_line_body = (
                -1
                if line_body == -1
                else affine_body_by_newton_body[line_body]
            )
            indices.append((affine_body_by_newton_body[point_body], affine_line_body))
            points.append(point.astype(np.float32))
            endpoints0.append(endpoint0.astype(np.float32))
            endpoints1.append(endpoint1.astype(np.float32))
            stiffnesses.append(stiffness)
            dampings.append(damping)

        return indices, points, endpoints0, endpoints1, stiffnesses, dampings

    @staticmethod
    def _validate_particle_world_order(worlds: list[int], world_count: int) -> None:
        """Validate particle world ids before world-local reordering."""
        if any(world < -1 or world >= world_count for world in worlds):
            raise ValueError(
                "Affine body proxy particles would receive an invalid Newton world index."
            )

    def _group_particles_by_world(self) -> None:
        """Stably group deformable particles and affine proxies by world."""
        if self.world_count <= 1 or not self.particle_q:
            return

        order = np.argsort(np.asarray(self.particle_world, dtype=np.int32), kind="stable")
        if np.array_equal(order, np.arange(len(order))):
            return
        old_to_new = np.empty(len(order), dtype=np.int32)
        old_to_new[order] = np.arange(len(order), dtype=np.int32)

        for name in (
            "particle_q",
            "particle_qd",
            "particle_mass",
            "particle_radius",
            "particle_flags",
            "particle_world",
        ):
            values = getattr(self, name)
            setattr(self, name, [values[index] for index in order])

        def remap(value: Any) -> Any:
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, tuple):
                return tuple(remap(item) for item in value)
            index = int(value)
            return index if index < 0 else int(old_to_new[index])

        for name in ("spring_indices", "tri_indices", "edge_indices", "tet_indices"):
            setattr(self, name, remap(getattr(self, name)))
        self.affine_body_particle_indices = [
            tuple(int(old_to_new[index]) for index in indices)
            for indices in self.affine_body_particle_indices
        ]

    def _dynamic_body_indices(self) -> list[int]:
        """Return bodies requiring contact or joint proxies in body order."""
        # Prescribed bodies need inactive proxies so their moving surfaces can
        # transmit contact forces to dynamic bodies.
        proxy_flags = int(_newton.BodyFlags.DYNAMIC | _newton.BodyFlags.KINEMATIC)
        return [
            body
            for body, flags in enumerate(self.body_flags)
            if int(flags) & proxy_flags
        ]

    def _add_affine_body_proxies(self) -> None:
        """Append proxy particles once all body mass properties are known."""
        if self._affine_body_proxies_added:
            return

        dynamic_bodies = self._dynamic_body_indices()
        proxy_worlds = [int(self.body_world[body]) for body in dynamic_bodies]
        self._validate_particle_world_order(
            [int(world) for world in self.particle_world] + proxy_worlds,
            self.world_count,
        )

        proxy_specs: list[
            tuple[
                int,
                int,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
        ] = []

        for body_index in dynamic_bodies:
            mass = float(self.body_mass[body_index])
            if not np.isfinite(mass) or mass < 0.0:
                raise ValueError(
                    f"Dynamic body {body_index} must have non-negative mass before finalization."
                )

            # Newton uses zero-mass dynamic bodies as prescribed cable endpoints.
            # Their inactive proxies still need a nonsingular native mass metric.
            proxy_mass = mass if mass > 0.0 else 1.0
            proxy_inertia = self.body_inertia[body_index] if mass > 0.0 else np.eye(3)
            mass_matrix = self._proxy_mass_matrix(
                body_index, proxy_mass, proxy_inertia
            )
            body_transform = np.asarray(self.body_q[body_index], dtype=np.float64)
            body_position = body_transform[:3]
            body_rotation = np.asarray(
                wp.quat_to_matrix(wp.quat(*body_transform[3:7])), dtype=np.float64
            ).reshape(3, 3)
            body_com = np.asarray(self.body_com[body_index], dtype=np.float64)
            body_velocity = np.asarray(self.body_qd[body_index], dtype=np.float64)
            world_com = body_position + body_rotation @ body_com
            linear_velocity = body_velocity[:3]
            angular_velocity = body_velocity[3:]
            affine_columns = body_rotation.T.copy()
            positions = np.vstack((world_com, affine_columns))
            velocities = np.vstack(
                (
                    linear_velocity,
                    [np.cross(angular_velocity, column) for column in affine_columns],
                )
            )

            proxy_specs.append(
                (
                    body_index,
                    int(self.body_world[body_index]),
                    positions,
                    velocities,
                    np.diag(mass_matrix).astype(np.float64),
                    mass_matrix,
                )
            )

        for (
            body_index,
            body_world,
            positions,
            velocities,
            masses,
            mass_matrix,
        ) in proxy_specs:
            particle_indices: list[int] = []
            for position, velocity, mass in zip(positions, velocities, masses):
                particle_index = self.add_particle(
                    pos=position,
                    vel=velocity,
                    mass=float(mass),
                    radius=self.affine_body_particle_radius,
                )
                # finalize() normally runs outside a world context, so preserve
                # the source body's assignment instead of inheriting -1 blindly.
                self.particle_world[particle_index] = body_world
                particle_indices.append(particle_index)

            self.affine_body_ids.append(body_index)
            self.affine_body_particle_indices.append(tuple(particle_indices))
            self.affine_body_particle_mass_matrices.append(mass_matrix)

        self._group_particles_by_world()
        self._affine_body_proxies_added = True

    def finalize(
        self,
        device: Any | None = None,
        *,
        requires_grad: bool = False,
        skip_all_validations: bool = False,
        skip_validation_worlds: bool = False,
        skip_validation_joints: bool = False,
        skip_validation_shapes: bool = False,
        skip_validation_structure: bool = False,
        skip_validation_joint_ordering: bool = True,
    ) -> Any:
        """Add affine-body proxies and finalize one Newton model.

        The returned model carries explicit body-id and particle-index arrays
        consumed by :class:`style3d_sim.NewtonSolver`. This builder is one-shot because
        appending bodies after proxy indices are fixed would invalidate them.
        """
        if self._synreal_model_finalized:
            raise RuntimeError(
                "style3d_sim.ModelBuilder can only be finalized once; create a new builder for another model."
            )

        self._add_affine_body_proxies()
        model = super().finalize(
            device=device,
            requires_grad=requires_grad,
            skip_all_validations=skip_all_validations,
            skip_validation_worlds=skip_validation_worlds,
            skip_validation_joints=skip_validation_joints,
            skip_validation_shapes=skip_validation_shapes,
            skip_validation_structure=skip_validation_structure,
            skip_validation_joint_ordering=skip_validation_joint_ordering,
        )

        body_ids = self.affine_body_ids
        (
            body_spring_indices,
            body_spring_points0,
            body_spring_points1,
            body_spring_stiffnesses,
            body_spring_rest_lengths,
        ) = self._finalize_body_springs(body_ids)
        (
            body_distance_constraint_indices,
            body_distance_constraint_points0,
            body_distance_constraint_points1,
            body_distance_constraint_weights0,
            body_distance_constraint_weights1,
            body_distance_constraint_rest_lengths,
        ) = self._finalize_body_distance_constraints(body_ids)
        (
            body_point_line_indices,
            body_point_line_points,
            body_point_line_endpoints0,
            body_point_line_endpoints1,
            body_point_line_stiffnesses,
            body_point_line_dampings,
        ) = self._finalize_body_point_lines(body_ids)
        particle_mass_diagonal = np.asarray(self.particle_mass, dtype=np.float32).copy()
        particle_mass_non_diagonal_indices: list[tuple[int, int]] = []
        particle_mass_non_diagonal_values: list[float] = []
        for particle_indices, mass_matrix in zip(
            self.affine_body_particle_indices, self.affine_body_particle_mass_matrices
        ):
            for local_index, particle_index in enumerate(particle_indices):
                particle_mass_diagonal[particle_index] = mass_matrix[
                    local_index, local_index
                ]
            for first, second in combinations(range(4), 2):
                value = float(mass_matrix[first, second])
                if value != 0.0:
                    particle_mass_non_diagonal_indices.append(
                        (particle_indices[first], particle_indices[second])
                    )
                    particle_mass_non_diagonal_values.append(value)

        model.synreal_particle_mass_diagonal = wp.array(
            particle_mass_diagonal, dtype=wp.float32, device=model.device
        )
        model.synreal_particle_mass_non_diagonal_indices = wp.array(
            particle_mass_non_diagonal_indices,
            dtype=wp.vec2i,
            device=model.device,
        )
        model.synreal_particle_mass_non_diagonal_values = wp.array(
            particle_mass_non_diagonal_values,
            dtype=wp.float32,
            device=model.device,
        )
        model.synreal_affine_body_ids = wp.array(
            body_ids, dtype=wp.int32, device=model.device
        )
        model.synreal_affine_body_indices = wp.array(
            self.affine_body_particle_indices, dtype=wp.vec4i, device=model.device
        )
        model.synreal_affine_body_masses = wp.array(
            [self.body_mass[body] if self.body_mass[body] > 0.0 else 1.0 for body in body_ids],
            dtype=wp.float32,
            device=model.device,
        )
        model.synreal_affine_body_inertia = wp.array(
            [np.asarray(self.body_inertia[body]).reshape(3, 3) if self.body_mass[body] > 0.0 else np.eye(3)
             for body in body_ids],
            dtype=wp.mat33,
            device=model.device,
        )
        model.synreal_affine_body_com = wp.array(
            [self.body_com[body] for body in body_ids],
            dtype=wp.vec3,
            device=model.device,
        )
        model.synreal_body_spring_indices = wp.array(
            body_spring_indices, dtype=wp.vec2i, device=model.device
        )
        model.synreal_body_spring_points0 = wp.array(
            body_spring_points0, dtype=wp.vec3, device=model.device
        )
        model.synreal_body_spring_points1 = wp.array(
            body_spring_points1, dtype=wp.vec3, device=model.device
        )
        model.synreal_body_spring_stiffnesses = wp.array(
            body_spring_stiffnesses, dtype=wp.float32, device=model.device
        )
        model.synreal_body_spring_rest_lengths = wp.array(
            body_spring_rest_lengths, dtype=wp.float32, device=model.device
        )
        model.synreal_body_distance_constraint_indices = wp.array(
            body_distance_constraint_indices, dtype=wp.vec2i, device=model.device
        )
        model.synreal_body_distance_constraint_points0 = wp.array(
            body_distance_constraint_points0, dtype=wp.vec3, device=model.device
        )
        model.synreal_body_distance_constraint_points1 = wp.array(
            body_distance_constraint_points1, dtype=wp.vec3, device=model.device
        )
        model.synreal_body_distance_constraint_weights0 = wp.array(
            body_distance_constraint_weights0, dtype=wp.vec4, device=model.device
        )
        model.synreal_body_distance_constraint_weights1 = wp.array(
            body_distance_constraint_weights1, dtype=wp.vec4, device=model.device
        )
        model.synreal_body_distance_constraint_rest_lengths = wp.array(
            body_distance_constraint_rest_lengths,
            dtype=wp.float32,
            device=model.device,
        )
        model.synreal_body_point_line_indices = wp.array(
            body_point_line_indices, dtype=wp.vec2i, device=model.device
        )
        model.synreal_body_point_line_points = wp.array(
            body_point_line_points, dtype=wp.vec3, device=model.device
        )
        model.synreal_body_point_line_endpoints0 = wp.array(
            body_point_line_endpoints0, dtype=wp.vec3, device=model.device
        )
        model.synreal_body_point_line_endpoints1 = wp.array(
            body_point_line_endpoints1, dtype=wp.vec3, device=model.device
        )
        model.synreal_body_point_line_stiffnesses = wp.array(
            body_point_line_stiffnesses, dtype=wp.float32, device=model.device
        )
        model.synreal_body_point_line_dampings = wp.array(
            body_point_line_dampings, dtype=wp.float32, device=model.device
        )
        self._synreal_model_finalized = True
        return model
