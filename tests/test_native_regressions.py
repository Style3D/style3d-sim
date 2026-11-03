import numpy as np
import pytest

import style3d_sim as s3d

wp = pytest.importorskip("warp")

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Native solver requires CUDA")


def _span(array: wp.array) -> s3d.Span:
    return s3d.Span(ptr=array.ptr, size=len(array))


def _state(positions: wp.array, velocities: wp.array) -> s3d.State:
    state = s3d.State()
    state.particle_q = _span(positions)
    state.particle_qd = _span(velocities)
    return state


@pytest.mark.parametrize(
    "nonlinear_solver", (s3d.NonlinearSolver.PD_NEWTON, s3d.NonlinearSolver.NEWTON)
)
@pytest.mark.parametrize("in_place", (False, True))
@pytest.mark.parametrize("capture", (False, True))
def test_zero_iterations_returns_inertial_prediction(
    nonlinear_solver, in_place: bool, capture: bool
) -> None:
    initial = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (2.0, 0.0, 3.0)), dtype=np.float32)
    velocities = np.array(((1.0, 2.0, 3.0),) * 3, dtype=np.float32)
    position_in = wp.array(initial, dtype=wp.vec3, device="cuda:0")
    velocity_in = wp.array(velocities, dtype=wp.vec3, device="cuda:0")
    position_out = position_in if in_place else wp.full(3, wp.vec3(-123.0), device="cuda:0")
    velocity_out = velocity_in if in_place else wp.zeros(3, dtype=wp.vec3, device="cuda:0")
    masses = wp.array((1.0, 0.0, 1.0), dtype=wp.float32, device="cuda:0")
    flags = wp.array((1, 1, 0), dtype=wp.int32, device="cuda:0")
    gravity = wp.array(((0.0, 0.0, -9.81),), dtype=wp.vec3, device="cuda:0")
    model = s3d.Model()
    model.particle_count = 3
    model.particle_masses = _span(masses)
    model.particle_flags = _span(flags)
    model.gravity = _span(gravity)
    solver = s3d.Solver()
    solver.initialize(model)
    solver.config.nonlinear_solver = nonlinear_solver
    solver.config.nonlinear_iterations = 0
    dt = 0.01
    state_in, state_out = _state(position_in, velocity_in), _state(position_out, velocity_out)
    if capture:
        with wp.ScopedCapture(device="cuda:0") as captured:
            solver.step(state_in, state_out, dt, wp.get_stream("cuda:0"))
        wp.capture_launch(captured.graph)
    else:
        solver.step(state_in, state_out, dt, wp.get_stream("cuda:0"))

    expected = initial.copy()
    expected[0] += dt * velocities[0] + dt * dt * gravity.numpy()[0]
    np.testing.assert_allclose(position_out.numpy(), expected, atol=1.0e-7)
    np.testing.assert_allclose(velocity_out.numpy(), (expected - initial) / dt, atol=1.0e-5)


@pytest.mark.parametrize("world_count", (1, 3))
def test_replicated_particles_contact_global_ground(world_count: int) -> None:
    newton = pytest.importorskip("newton")
    scene = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    scene.add_particle(pos=(0.0, 0.0, 0.05), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    builder.replicate(scene, world_count)
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model)
    if world_count > 1:
        assert solver.synreal_model.world_count == world_count
        assert solver.synreal_model.shape_replicated_count == 0
    state_in, state_out = model.state(), model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in.clear_forces()
    pipeline.collide(state_in, contacts)
    assert int(contacts.soft_contact_count.numpy()[0]) == world_count
    solver.step(state_in, state_out, None, contacts, 0.01)

    positions = state_out.particle_q.numpy()
    assert np.all(positions[:, 2] > 0.05 + 1.0e-5)
    np.testing.assert_allclose(
        positions, np.repeat(positions[:1], world_count, axis=0), atol=1.0e-6
    )


@pytest.mark.parametrize(
    "nonlinear_solver", (s3d.NonlinearSolver.PD_NEWTON, s3d.NonlinearSolver.NEWTON)
)
@pytest.mark.parametrize("fixed_side", ("face", "particle", "both"))
@pytest.mark.parametrize("fixed_by_flags", (False, True))
def test_vertex_face_contact_with_fixed_particles(
    nonlinear_solver, fixed_side: str, fixed_by_flags: bool
) -> None:
    initial = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.25, 0.25, 0.04)),
        dtype=np.float32,
    )
    fixed = np.array((fixed_side != "particle",) * 3 + (fixed_side != "face",))
    mass_values = np.ones(4, dtype=np.float32)
    flag_values = np.ones(4, dtype=np.int32)
    if fixed_by_flags:
        flag_values[fixed] = 0
    else:
        mass_values[fixed] = 0.0
    masses = wp.array(mass_values, dtype=wp.float32, device="cuda:0")
    flags = wp.array(flag_values, dtype=wp.int32, device="cuda:0")
    radii = wp.full(4, 0.08, dtype=wp.float32, device="cuda:0")
    triangles = wp.array(((0, 1, 2),), dtype=wp.vec3i, device="cuda:0")
    particles = wp.array((3,), dtype=wp.int32, device="cuda:0")
    position_in = wp.array(initial, dtype=wp.vec3, device="cuda:0")
    velocity_in = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
    position_out = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
    velocity_out = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
    model = s3d.Model()
    model.particle_count = 4
    model.particle_masses = _span(masses)
    model.particle_flags = _span(flags)
    model.particle_radii = _span(radii)
    model.particle_rest_positions = _span(position_in)
    model.collision_triangle_count = 1
    model.collision_triangle_indices = _span(triangles)
    model.collision_particle_count = 1
    model.collision_particle_indices = _span(particles)
    solver = s3d.Solver()
    solver.initialize(model)
    solver.config.nonlinear_solver = nonlinear_solver
    solver.config.enable_self_collision = True
    solver.config.self_collision_radius = 0.02
    solver.config.self_collision_vertex_face_stiffness = 2.0
    solver.step(_state(position_in, velocity_in), _state(position_out, velocity_out), 0.01)

    result = position_out.numpy()
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[fixed], initial[fixed], atol=1.0e-7)
    if fixed_side == "face":
        assert result[3, 2] > initial[3, 2] + 1.0e-5
    elif fixed_side == "particle":
        assert np.mean(result[:3, 2]) < -1.0e-5


@pytest.mark.parametrize("mass", (1.0, 1.0e16))
def test_particle_collision_stiffness_remains_finite_for_large_mass(mass: float) -> None:
    position_in = wp.array(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)), dtype=wp.vec3, device="cuda:0")
    velocity_in = wp.zeros(2, dtype=wp.vec3, device="cuda:0")
    position_out = wp.zeros(2, dtype=wp.vec3, device="cuda:0")
    velocity_out = wp.zeros(2, dtype=wp.vec3, device="cuda:0")
    masses = wp.full(2, mass, dtype=wp.float32, device="cuda:0")
    radii = wp.full(2, 0.1, dtype=wp.float32, device="cuda:0")
    particles = wp.array((0, 1), dtype=wp.int32, device="cuda:0")
    model = s3d.Model()
    model.particle_count = 2
    model.particle_masses = _span(masses)
    model.particle_radii = _span(radii)
    model.particle_rest_positions = _span(position_in)
    model.collision_particle_count = 2
    model.collision_particle_indices = _span(particles)
    solver = s3d.Solver()
    solver.initialize(model)
    solver.config.enable_self_collision = True
    solver.step(_state(position_in, velocity_in), _state(position_out, velocity_out), 0.01)

    result = position_out.numpy()
    assert np.isfinite(result).all()
    assert np.isfinite(velocity_out.numpy()).all()
    assert result[1, 0] - result[0, 0] > 0.1 + 1.0e-5
    np.testing.assert_allclose(result.mean(axis=0), (0.05, 0.0, 0.0), atol=1.0e-6)
