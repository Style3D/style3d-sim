"""Newton 1.7 solver-observable integration tests."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

import style3d_sim as s3d  # noqa: E402


pytestmark = pytest.mark.skipif(
    not wp.is_cuda_available(), reason="NewtonSolver observable tests require CUDA"
)


def _step_with_observables(
    model: object,
    solver: s3d.NewtonSolver,
    pipeline: object,
    contacts: object,
    kinds: set[str],
    *,
    dt: float = 0.005,
) -> tuple[object, object, object]:
    state_in, state_out = model.state(), model.state()
    state_in.clear_forces()
    pipeline.collide(state_in, contacts)
    observables = solver.observables(kinds, contacts=contacts)
    solver.step(
        state_in,
        state_out,
        model.control(),
        contacts,
        dt,
        observables=observables,
    )
    wp.synchronize_device(model.device)
    return state_in, state_out, observables


def test_body_qdd_observable_covers_replicated_body_states() -> None:
    scene = s3d.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    body = scene.add_body(
        xform=wp.transform(wp.vec3(0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
    )
    scene.body_qd[body] = wp.spatial_vector(0.0)
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.replicate(scene, 2)
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model)
    state_in, state_out = model.state(), model.state()
    observables = solver.observables({solver.ObservableKind.BODY_QDD})

    solver.step(state_in, state_out, None, None, 0.005, observables=observables)
    wp.synchronize_device(model.device)

    expected = (state_out.body_qd.numpy() - state_in.body_qd.numpy()) / 0.005
    actual = observables.body_qdd.numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=2.0e-4, rtol=2.0e-4)
    assert np.linalg.norm(actual[1, :3]) > 1.0

    with pytest.raises(ValueError, match="does not support"):
        solver.observables({solver.ObservableKind.BODY_PARENT_F})


@pytest.mark.parametrize("hard_contacts", (False, True), ids=("penalty", "hard"))
def test_contact_f_observable_reports_rigid_pair_wrench_and_clears_unused_rows(
    hard_contacts: bool,
) -> None:
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 100.0
    builder.default_shape_cfg.mu = 0.0
    builder.add_ground_plane()
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.24), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body, radius=0.25)
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model)
    solver.config.rigid_contact_hard = hard_contacts
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    _, _, observables = _step_with_observables(
        model,
        solver,
        pipeline,
        contacts,
        {solver.ObservableKind.CONTACT_F},
    )

    rigid_capacity = int(contacts.rigid_contact_max)
    rigid_count = int(contacts.rigid_contact_count.numpy()[0])
    assert rigid_count > 0
    forces = observables.contact_f.numpy()
    shape0 = int(contacts.rigid_contact_shape0.numpy()[0])
    body0 = int(model.shape_body.numpy()[shape0])
    force_on_body0 = 1.0 if body0 == body else -1.0
    assert force_on_body0 * forces[0, 2] > 0.0
    assert np.linalg.norm(forces[0, :3]) > 1.0e-3
    np.testing.assert_allclose(forces[rigid_count:rigid_capacity], 0.0, atol=0.0)
    np.testing.assert_allclose(forces[rigid_capacity:], 0.0, atol=0.0)


def test_contact_f_observable_reports_soft_particle_wrench_about_world_origin() -> None:
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.add_ground_plane()
    builder.add_particle(pos=(0.2, 0.0, 0.04), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05)
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke = 1.0e5
    model.soft_contact_kd = 0.0
    solver = s3d.NewtonSolver(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    _, _, observables = _step_with_observables(
        model,
        solver,
        pipeline,
        contacts,
        {solver.ObservableKind.CONTACT_F},
    )

    rigid_capacity = int(contacts.rigid_contact_max)
    soft_count = int(contacts.soft_contact_count.numpy()[0])
    assert soft_count > 0
    force = observables.contact_f.numpy()[rigid_capacity]
    assert force[2] < 0.0
    contact_point = contacts.soft_contact_body_pos.numpy()[0]
    np.testing.assert_allclose(force[3:], np.cross(contact_point, force[:3]), atol=1.0e-3, rtol=1.0e-3)
    np.testing.assert_allclose(observables.contact_f.numpy()[rigid_capacity + soft_count :], 0.0, atol=0.0)


def test_contact_f_observable_reports_full_surface_soft_contacts() -> None:
    from newton.solvers import style3d

    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(-0.05, -0.05, 0.01),
        rot=wp.quat_identity(),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        vel=wp.vec3(0.0),
        mass=0.1,
        particle_radius=0.02,
    )
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 0.0
    solver = s3d.NewtonSolver(model)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()

    _, _, observables = _step_with_observables(
        model,
        solver,
        pipeline,
        contacts,
        {solver.ObservableKind.CONTACT_F},
    )

    rigid_capacity = int(contacts.rigid_contact_max)
    soft_count = int(contacts.soft_contact_count.numpy()[0])
    assert soft_count > 0
    forces = observables.contact_f.numpy()[rigid_capacity : rigid_capacity + soft_count]
    assert np.isfinite(forces).all()
    assert np.any(np.linalg.norm(forces[:, :3], axis=1) > 0.0)


def test_contact_f_observable_includes_rolling_resistance_torque() -> None:
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.default_shape_cfg.density = 0.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 100.0
    builder.default_shape_cfg.kf = 1.0e4
    builder.default_shape_cfg.mu = 0.0
    builder.default_shape_cfg.mu_rolling = 0.1
    builder.default_shape_cfg.mu_torsional = 0.0
    builder.add_ground_plane()
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.249), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.025, 0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.025),
        lock_inertia=True,
    )
    builder.body_qd[body] = wp.spatial_vector(0.5, 0.0, 0.0, 0.0, 2.0, 0.0)
    builder.add_shape_sphere(body, radius=0.25)
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    _, _, observables = _step_with_observables(
        model,
        solver,
        pipeline,
        contacts,
        {solver.ObservableKind.CONTACT_F},
        dt=0.002,
    )

    rigid_count = int(contacts.rigid_contact_count.numpy()[0])
    assert rigid_count > 0
    forces = observables.contact_f.numpy()[:rigid_count]
    assert np.max(np.linalg.norm(forces[:, 3:], axis=1)) > 1.0e-4


def test_solver_observables_support_cuda_graph_capture_and_replay() -> None:
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 100.0
    builder.add_ground_plane()
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.20), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body, radius=0.25)
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    observables = solver.observables(
        {solver.ObservableKind.BODY_QDD, solver.ObservableKind.CONTACT_F},
        contacts=contacts,
    )

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, None, contacts, 0.005, observables=observables)
    wp.synchronize_device(model.device)
    pipeline.collide(state_out, contacts)
    assert int(contacts.rigid_contact_count.numpy()[0]) > 0
    observables.contact_f.fill_(1.0)
    wp.synchronize_device(model.device)
    stream = wp.get_stream(model.device)
    with wp.ScopedCapture(stream=stream) as capture:
        solver.step(state_out, state_in, None, contacts, 0.005, observables=observables)

    wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    expected_qdd = (state_in.body_qd.numpy() - state_out.body_qd.numpy()) / 0.005
    np.testing.assert_allclose(observables.body_qdd.numpy(), expected_qdd, atol=2.0e-4, rtol=2.0e-4)
    np.testing.assert_allclose(observables.contact_f.numpy(), 0.0, atol=0.0)
