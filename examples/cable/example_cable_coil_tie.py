# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Coil with Tie
#
# Five cables of different thicknesses are wound into ten stacked turns, then
# held together by a free, closed cable loop acting as a tie.
#
# Run interactively:
#   uv run --extra examples python examples/cable/example_cable_coil_tie.py
#
# Run as a test:
#   uv run --extra examples python examples/cable/example_cable_coil_tie.py --test --viewer null
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

try:
    from examples.cable.cable_utils import (
        create_cable_builder,
        create_cable_parser,
        create_cable_solver,
        update_cable_contact_history,
    )
except ModuleNotFoundError:
    from cable_utils import create_cable_builder, create_cable_parser, create_cable_solver, update_cable_contact_history


class Example:
    NUM_LOOPS = 10
    SEGMENTS_PER_LOOP = 32
    COIL_RADIUS = 0.35
    COIL_RADIAL_OFFSETS = (-0.070, -0.035, 0.0, 0.035, 0.070)
    COIL_CABLE_RADII = (0.010, 0.014, 0.018, 0.012, 0.009)
    TURN_PITCH = 0.032
    COIL_BOTTOM_Z = 0.065
    TIE_SEGMENTS = 80
    TIE_CABLE_RADIUS = 0.008

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = create_cable_builder(args, gravity=(0.0, 0.0, -9.8))
        builder.rigid_gap = 0.0
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 0.8
        builder.add_ground_plane()

        self.coil_bodies = []
        for strand_index, (radial_offset, cable_radius) in enumerate(
            zip(self.COIL_RADIAL_OFFSETS, self.COIL_CABLE_RADII, strict=True)
        ):
            coil_points = self._make_coil_points(radial_offset)
            coil_rod = newton.Rod(coil_points, radius=cable_radius)
            strand_bodies, _ = builder.add_rod(
                rod=coil_rod,
                stretch_stiffness=1.0e6,
                bend_stiffness=2.0e2,
                twist_stiffness=2.0e2,
                bend_damping=2.0,
                twist_damping=2.0,
                label=f"ten_turn_coil_{strand_index}",
                body_frame_origin="com",
            )
            self.coil_bodies.extend(strand_bodies)

        tie_points = self._make_tie_points()
        tie_rod = newton.Rod(tie_points, radius=self.TIE_CABLE_RADIUS, closed=True)
        self.tie_bodies, _ = builder.add_rod(
            rod=tie_rod,
            stretch_stiffness=1.0e6,
            bend_stiffness=4.0e2,
            twist_stiffness=4.0e2,
            bend_damping=2.0,
            twist_damping=2.0,
            label="cable_tie",
            body_frame_origin="com",
        )

        builder.color()
        self.model = builder.finalize()
        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.collision_pipeline.contacts()
        self.solver = create_cable_solver(
            args,
            self.model,
            iterations=self.sim_iterations,
            rigid_contact_hard=True,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=2048,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.9, -1.1, 0.65), pitch=0.5, yaw=-35.0)
            if hasattr(self.viewer, "camera"):
                self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.20))
                self.viewer.camera.fov = 40.0

        self.capture()

    def _make_coil_points(self, radial_offset: float) -> list[wp.vec3]:
        num_segments = self.NUM_LOOPS * self.SEGMENTS_PER_LOOP
        angles = np.linspace(0.0, 2.0 * np.pi * self.NUM_LOOPS, num_segments + 1)
        radius = self.COIL_RADIUS + radial_offset
        points = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = self.COIL_BOTTOM_Z + self.TURN_PITCH * angle / (2.0 * np.pi)
            points.append(wp.vec3(float(x), float(y), float(z)))
        return points

    def _make_tie_points(self) -> list[wp.vec3]:
        half_turn_span = 0.5 * (self.NUM_LOOPS - 1) * self.TURN_PITCH
        coil_center_z = self.COIL_BOTTOM_Z + half_turn_span
        coil_outer_radius = max(
            np.hypot(abs(offset), half_turn_span + cable_radius) + cable_radius
            for offset, cable_radius in zip(self.COIL_RADIAL_OFFSETS, self.COIL_CABLE_RADII, strict=True)
        )
        tie_radius = coil_outer_radius + self.TIE_CABLE_RADIUS - 0.001
        angles = np.linspace(0.0, 2.0 * np.pi, self.TIE_SEGMENTS, endpoint=False)
        points = [
            wp.vec3(
                float(self.COIL_RADIUS + tie_radius * np.cos(angle)),
                0.0,
                float(coil_center_z + tie_radius * np.sin(angle)),
            )
            for angle in angles
        ]
        points.append(points[0])
        return points

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            update_cable_contact_history(self.solver, True)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise ValueError("Coiled cable transforms contain non-finite values.")

        expected_coil_segments = self.NUM_LOOPS * self.SEGMENTS_PER_LOOP * len(self.COIL_CABLE_RADII)
        if len(self.coil_bodies) != expected_coil_segments:
            raise ValueError(f"Expected {expected_coil_segments} coil segments, found {len(self.coil_bodies)}.")

        coil_positions = body_q[self.coil_bodies, :3]
        if np.max(np.linalg.norm(coil_positions, axis=1)) > 2.0:
            raise ValueError("Coiled cable moved outside the expected scene bounds.")

    @staticmethod
    def create_parser():
        return create_cable_parser(Path(__file__).stem)


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
