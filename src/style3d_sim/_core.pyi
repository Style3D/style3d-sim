# SPDX-FileCopyrightText: Copyright (c) 2026 Zhejiang Linctex Digital Technology Ltd.
# SPDX-License-Identifier: Apache-2.0

"""

Native bindings for Style3D Sim.

Array fields are represented by non-owning Span objects that refer to CUDA
device memory. The caller is responsible for keeping every referenced device
allocation alive for as long as native code may access it.
"""
from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['AffineBodyRigidityModel', 'Config', 'Contact', 'Control', 'InertiaMode', 'JointConstraintMode', 'LogLevel', 'Model', 'NonlinearSolver', 'Solver', 'Span', 'State', 'is_logged_in', 'login', 'set_log_callback']
class AffineBodyRigidityModel:
    """
    Members:

      CO_ROTATED

      ABD_ORTHOGONALITY

      SIX_SPRINGS

      SO3_AUGMENTED_LAGRANGIAN
    """
    ABD_ORTHOGONALITY: typing.ClassVar[AffineBodyRigidityModel]  # value = <AffineBodyRigidityModel.ABD_ORTHOGONALITY: 1>
    CO_ROTATED: typing.ClassVar[AffineBodyRigidityModel]  # value = <AffineBodyRigidityModel.CO_ROTATED: 0>
    SIX_SPRINGS: typing.ClassVar[AffineBodyRigidityModel]  # value = <AffineBodyRigidityModel.SIX_SPRINGS: 2>
    SO3_AUGMENTED_LAGRANGIAN: typing.ClassVar[AffineBodyRigidityModel]  # value = <AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN: 3>
    __members__: typing.ClassVar[dict[str, AffineBodyRigidityModel]]  # value = {'CO_ROTATED': <AffineBodyRigidityModel.CO_ROTATED: 0>, 'ABD_ORTHOGONALITY': <AffineBodyRigidityModel.ABD_ORTHOGONALITY: 1>, 'SIX_SPRINGS': <AffineBodyRigidityModel.SIX_SPRINGS: 2>, 'SO3_AUGMENTED_LAGRANGIAN': <AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN: 3>}
    @typing.overload
    def __eq__(self, other: AffineBodyRigidityModel) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: AffineBodyRigidityModel) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Config:
    """

    Runtime configuration for a Style3D Sim solver.

    Changes made through ``solver.config`` affect subsequent direct step calls.
    Previously captured CUDA graphs must be rebuilt after a configuration change.
    """
    def __init__(self) -> None:
        """
        Create the default solver configuration.
        """
    @property
    def affine_body_block_preconditioner(self) -> bool:
        """Use a full affine-body block inverse as the PCG preconditioner."""
    @affine_body_block_preconditioner.setter
    def affine_body_block_preconditioner(self, arg0: bool) -> None:
        ...
    @property
    def affine_body_contact_block_preconditioner(self) -> bool:
        """Include hard-contact cross terms in affine-body PCG blocks; the contact operator is unchanged."""
    @affine_body_contact_block_preconditioner.setter
    def affine_body_contact_block_preconditioner(self, arg0: bool) -> None:
        ...
    @property
    def affine_body_rigidity_model(self) -> AffineBodyRigidityModel:
        """
        Rigidity law used for affine bodies: co-rotated, original ABD orthogonality, six springs, or SO(3) AL.
        """
    @affine_body_rigidity_model.setter
    def affine_body_rigidity_model(self, arg0: AffineBodyRigidityModel) -> None:
        ...
    @property
    def affine_body_rigidity_primal_iterations(self) -> int:
        """
        Primal Newton iterations performed with a fixed affine-body SO(3) AL state before each update.
        """
    @affine_body_rigidity_primal_iterations.setter
    def affine_body_rigidity_primal_iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def air_density(self) -> float:
        """
        Ambient air density in kg/m^3; zero disables aerodynamic drag.
        """
    @air_density.setter
    def air_density(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def constraint_alpha(self) -> float:
        """
        Shared dimensionless mass-normalized penalty for affine rigidity, generated joint constraints, and hard distance constraints. Zero disables all three.
        """
    @constraint_alpha.setter
    def constraint_alpha(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def enable_collision_hessian_non_diagonal(self) -> bool:
        """
        Include coupled contact tangents for rigid, particle-shape, and all self-collision contacts. False keeps their diagonal approximation; full-surface and angular-friction tangents remain coupled.
        """
    @enable_collision_hessian_non_diagonal.setter
    def enable_collision_hessian_non_diagonal(self, arg0: bool) -> None:
        ...
    @property
    def enable_self_collision(self) -> bool:
        """
        Whether deformable self-collision is enabled.
        """
    @enable_self_collision.setter
    def enable_self_collision(self, arg0: bool) -> None:
        ...
    @property
    def fluid_max_step_ratio(self) -> float:
        """
        Bound fluid Newton increments relative to kernel radius; zero disables.
        """
    @fluid_max_step_ratio.setter
    def fluid_max_step_ratio(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fluid_relaxation(self) -> float:
        """
        Global Newton step scale in fluid scenes, in (0, 1].
        """
    @fluid_relaxation.setter
    def fluid_relaxation(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def inertia_mode(self) -> InertiaMode:
        """
        ABD inertia construction and velocity recovery mode; NONE selects the free-particle path.
        """
    @inertia_mode.setter
    def inertia_mode(self, arg0: InertiaMode) -> None:
        ...
    @property
    def joint_constraint_mode(self) -> JointConstraintMode:
        """
        Joint-constraint enforcement method; PENALTY preserves the legacy behavior.
        """
    @joint_constraint_mode.setter
    def joint_constraint_mode(self, arg0: JointConstraintMode) -> None:
        ...
    @property
    def joint_constraint_primal_iterations(self) -> int:
        """
        Primal Newton iterations performed with a fixed augmented-Lagrangian dual before each multiplier update.
        """
    @joint_constraint_primal_iterations.setter
    def joint_constraint_primal_iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def linear_iterations(self) -> int:
        """
        PCG iterations per nonlinear iteration.
        """
    @linear_iterations.setter
    def linear_iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nonlinear_iterations(self) -> int:
        """
        Nonlinear iterations per step.
        """
    @nonlinear_iterations.setter
    def nonlinear_iterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nonlinear_solver(self) -> NonlinearSolver:
        """
        Implicit-Euler nonlinear method: NEWTON or the legacy PD_NEWTON method.
        """
    @nonlinear_solver.setter
    def nonlinear_solver(self, arg0: NonlinearSolver) -> None:
        ...
    @property
    def rolling_friction_velocity_epsilon(self) -> float:
        """
        Positive contact angular-speed regularization for rolling and torsional friction [rad/s].
        """
    @rolling_friction_velocity_epsilon.setter
    def rolling_friction_velocity_epsilon(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rigid_contact_friction_epsilon(self) -> float:
        """
        VBD-style rigid Coulomb smoothing speed [m/s]; zero preserves the native secant law.
        """
    @rigid_contact_friction_epsilon.setter
    def rigid_contact_friction_epsilon(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rigid_contact_hard(self) -> bool:
        """Use augmented-Lagrangian normal contact and displacement-based Coulomb friction."""
    @rigid_contact_hard.setter
    def rigid_contact_hard(self, arg0: bool) -> None:
        ...
    @property
    def rigid_contact_history(self) -> bool:
        """Warm-start hard-contact multipliers through Contact.match_index."""
    @rigid_contact_history.setter
    def rigid_contact_history(self, arg0: bool) -> None:
        ...
    @property
    def rigid_contact_sliding_tangent(self) -> bool:
        """Experimental Coulomb disk tangent; keep False for validated profiles because coupled ADMM can diverge."""
    @rigid_contact_sliding_tangent.setter
    def rigid_contact_sliding_tangent(self, arg0: bool) -> None:
        ...
    @property
    def rigid_contact_alpha(self) -> float:
        """Hard-contact C0 stabilization in [0, 1]."""
    @rigid_contact_alpha.setter
    def rigid_contact_alpha(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rigid_contact_beta(self) -> float:
        """Hard-contact penalty ramp per iteration."""
    @rigid_contact_beta.setter
    def rigid_contact_beta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rigid_contact_gamma(self) -> float:
        """Hard-contact history decay in [0, 1]."""
    @rigid_contact_gamma.setter
    def rigid_contact_gamma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rigid_contact_k_start(self) -> float:
        """Initial hard-contact penalty when ramping is enabled."""
    @rigid_contact_k_start.setter
    def rigid_contact_k_start(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cable_linear_beta(self) -> float:
        """Stretch/shear penalty ramp per nonlinear iteration; zero uses authored stiffness."""
    @cable_linear_beta.setter
    def cable_linear_beta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cable_angular_beta(self) -> float:
        """Bend/twist penalty ramp per nonlinear iteration; zero uses authored stiffness."""
    @cable_angular_beta.setter
    def cable_angular_beta(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cable_penalty_gamma(self) -> float:
        """Per-step cable penalty decay, bounded by the configured seed and authored cap."""
    @cable_penalty_gamma.setter
    def cable_penalty_gamma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cable_linear_k_start(self) -> float:
        """Stretch/shear penalty seed used only when cable_linear_beta is positive."""
    @cable_linear_k_start.setter
    def cable_linear_k_start(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cable_angular_k_start(self) -> float:
        """Bend/twist penalty seed used only when cable_angular_beta is positive."""
    @cable_angular_k_start.setter
    def cable_angular_k_start(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def self_collision_edge_edge_stiffness(self) -> float:
        """
        Dimensionless stiffness scale for edge-edge self-collision penalties.
        """
    @self_collision_edge_edge_stiffness.setter
    def self_collision_edge_edge_stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def self_collision_edge_face_stiffness(self) -> float:
        """
        Dimensionless stiffness scale for edge-face untangling penalties.
        """
    @self_collision_edge_face_stiffness.setter
    def self_collision_edge_face_stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def self_collision_particle_particle_stiffness(self) -> float:
        """
        Dimensionless stiffness scale for free-particle pair collision penalties.
        """
    @self_collision_particle_particle_stiffness.setter
    def self_collision_particle_particle_stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def self_collision_radius(self) -> float:
        """
        Radius used to thicken deformable collision primitives.
        """
    @self_collision_radius.setter
    def self_collision_radius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def self_collision_vertex_face_stiffness(self) -> float:
        """
        Dimensionless stiffness scale for vertex-face self-collision penalties.
        """
    @self_collision_vertex_face_stiffness.setter
    def self_collision_vertex_face_stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wind_velocity(self) -> tuple[float, float, float]:
        """
        Ambient air velocity as a world-coordinate (x, y, z) tuple in m/s.
        """
    @wind_velocity.setter
    def wind_velocity(self, arg1: collections.abc.Sequence) -> None:
        ...
class Contact:
    """

    Non-owning views of Newton rigid- and soft-contact buffers consumed by the native solver.
    """
    def __init__(self) -> None:
        """
        Create an empty contact view.
        """
    @property
    def count(self) -> Span:
        """
        Device span containing the active rigid-contact count.
        """
    @count.setter
    def count(self, arg1: Span) -> None:
        ...
    @property
    def history_refresh(self) -> bool:
        """False reuses current contact rows between collision refreshes."""
    @history_refresh.setter
    def history_refresh(self, arg0: bool) -> None:
        ...
    @property
    def match_index(self) -> Span:
        """Optional current-to-previous collision row correspondence for contact warm starts."""
    @match_index.setter
    def match_index(self, arg1: Span) -> None:
        ...
    @property
    def contact_force(self) -> Span:
        """Optional wrench output, rigid rows followed by soft rows."""
    @contact_force.setter
    def contact_force(self, arg1: Span) -> None:
        ...
    @property
    def margin0(self) -> Span:
        """
        Device span containing the first shape contact margin.
        """
    @margin0.setter
    def margin0(self, arg1: Span) -> None:
        ...
    @property
    def margin1(self) -> Span:
        """
        Device span containing the second shape contact margin.
        """
    @margin1.setter
    def margin1(self, arg1: Span) -> None:
        ...
    @property
    def max_count(self) -> int:
        """
        Maximum number of rigid contacts stored in the arrays.
        """
    @max_count.setter
    def max_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def normal(self) -> Span:
        """
        Device span containing world-space normals from shape zero to shape one.
        """
    @normal.setter
    def normal(self, arg1: Span) -> None:
        ...
    @property
    def offset0(self) -> Span:
        """
        Optional body-frame radial offset from the first contact point to its friction surface anchor.
        """
    @offset0.setter
    def offset0(self, arg1: Span) -> None:
        ...
    @property
    def offset1(self) -> Span:
        """
        Optional body-frame radial offset from the second contact point to its friction surface anchor.
        """
    @offset1.setter
    def offset1(self, arg1: Span) -> None:
        ...
    @property
    def point0(self) -> Span:
        """
        Device span containing body-frame points on the first shape.
        """
    @point0.setter
    def point0(self, arg1: Span) -> None:
        ...
    @property
    def point1(self) -> Span:
        """
        Device span containing body-frame points on the second shape.
        """
    @point1.setter
    def point1(self, arg1: Span) -> None:
        ...
    @property
    def shape0(self) -> Span:
        """
        Device span containing the first shape index.
        """
    @shape0.setter
    def shape0(self, arg1: Span) -> None:
        ...
    @property
    def shape1(self) -> Span:
        """
        Device span containing the second shape index.
        """
    @shape1.setter
    def shape1(self, arg1: Span) -> None:
        ...
    @property
    def soft_barycentric(self) -> Span:
        """
        Optional barycentric weights for soft_indices; replaces the single-particle contact view.
        """
    @soft_barycentric.setter
    def soft_barycentric(self, arg1: Span) -> None:
        ...
    @property
    def soft_body_pos(self) -> Span:
        """
        Device span containing body-local contact points on collision shapes.
        """
    @soft_body_pos.setter
    def soft_body_pos(self, arg1: Span) -> None:
        ...
    @property
    def soft_body_vel(self) -> Span:
        """
        Device span containing body-local surface velocities at soft contacts.
        """
    @soft_body_vel.setter
    def soft_body_vel(self, arg1: Span) -> None:
        ...
    @property
    def soft_count(self) -> Span:
        """
        Device span containing the active soft-contact count.
        """
    @soft_count.setter
    def soft_count(self, arg1: Span) -> None:
        ...
    @property
    def soft_indices(self) -> Span:
        """
        Optional particle, edge, or face contact ids as -1-padded triples; paired with soft_barycentric.
        """
    @soft_indices.setter
    def soft_indices(self, arg1: Span) -> None:
        ...
    @property
    def soft_max_count(self) -> int:
        """
        Maximum number of soft contacts stored in the arrays.
        """
    @soft_max_count.setter
    def soft_max_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def soft_normal(self) -> Span:
        """
        Device span containing world-space normals from shapes to particles.
        """
    @soft_normal.setter
    def soft_normal(self, arg1: Span) -> None:
        ...
    @property
    def soft_particle(self) -> Span:
        """
        Device span containing the particle index of each soft contact.
        """
    @soft_particle.setter
    def soft_particle(self, arg1: Span) -> None:
        ...
    @property
    def soft_particle_only_shapes(self) -> Span:
        """
        Optional per-original-shape flags selecting legacy particle contacts instead of full-surface contacts.
        """
    @soft_particle_only_shapes.setter
    def soft_particle_only_shapes(self, arg1: Span) -> None:
        ...
    @property
    def soft_shape(self) -> Span:
        """
        Device span containing the shape index of each soft contact.
        """
    @soft_shape.setter
    def soft_shape(self, arg1: Span) -> None:
        ...
class Control:
    """

    Non-owning views of Newton joint-control buffers consumed by the native solver.
    """
    def __init__(self) -> None:
        """
        Create an empty control view.
        """
    @property
    def joint_f(self) -> Span:
        """
        Device span containing generalized feedforward joint forces.
        """
    @joint_f.setter
    def joint_f(self, arg1: Span) -> None:
        ...
    @property
    def joint_drive_velocities(self) -> Span:
        """Optional beginning-of-step scalar joint velocities, indexed by drive state."""
    @joint_drive_velocities.setter
    def joint_drive_velocities(self, arg1: Span) -> None:
        ...
    @property
    def joint_target_q(self) -> Span:
        """
        Device span containing generalized joint-position targets.
        """
    @joint_target_q.setter
    def joint_target_q(self, arg1: Span) -> None:
        ...
    @property
    def joint_target_qd(self) -> Span:
        """
        Device span containing generalized joint-velocity targets.
        """
    @joint_target_qd.setter
    def joint_target_qd(self, arg1: Span) -> None:
        ...
    @property
    def tetrahedron_activations(self) -> Span:
        """
        Device span containing optional per-tetrahedron volumetric activations.
        """
    @tetrahedron_activations.setter
    def tetrahedron_activations(self, arg1: Span) -> None:
        ...
class InertiaMode:
    """
    Members:

      EXPLICIT_EULER

      IMPLICIT_EULER

      LIE_GROUP_MIDPOINT

      MIDPOINT_CAYLEY

      NONE

      VBD_SEMI_IMPLICIT
    """
    EXPLICIT_EULER: typing.ClassVar[InertiaMode]  # value = <InertiaMode.EXPLICIT_EULER: 0>
    IMPLICIT_EULER: typing.ClassVar[InertiaMode]  # value = <InertiaMode.IMPLICIT_EULER: 1>
    LIE_GROUP_MIDPOINT: typing.ClassVar[InertiaMode]  # value = <InertiaMode.LIE_GROUP_MIDPOINT: 2>
    MIDPOINT_CAYLEY: typing.ClassVar[InertiaMode]  # value = <InertiaMode.MIDPOINT_CAYLEY: 3>
    NONE: typing.ClassVar[InertiaMode]  # value = <InertiaMode.NONE: 4>
    VBD_SEMI_IMPLICIT: typing.ClassVar[InertiaMode]  # value = <InertiaMode.VBD_SEMI_IMPLICIT: 5>
    __members__: typing.ClassVar[dict[str, InertiaMode]]
    @typing.overload
    def __eq__(self, other: InertiaMode) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: InertiaMode) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class JointConstraintMode:
    """
    Members:

      PENALTY

      AUGMENTED_LAGRANGIAN
    """
    AUGMENTED_LAGRANGIAN: typing.ClassVar[JointConstraintMode]  # value = <JointConstraintMode.AUGMENTED_LAGRANGIAN: 1>
    PENALTY: typing.ClassVar[JointConstraintMode]  # value = <JointConstraintMode.PENALTY: 0>
    __members__: typing.ClassVar[dict[str, JointConstraintMode]]  # value = {'PENALTY': <JointConstraintMode.PENALTY: 0>, 'AUGMENTED_LAGRANGIAN': <JointConstraintMode.AUGMENTED_LAGRANGIAN: 1>}
    @typing.overload
    def __eq__(self, other: JointConstraintMode) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: JointConstraintMode) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LogLevel:
    """
    Members:

      INFO

      ERROR

      DEBUG

      WARNING
    """
    DEBUG: typing.ClassVar[LogLevel]  # value = <LogLevel.DEBUG: 2>
    ERROR: typing.ClassVar[LogLevel]  # value = <LogLevel.ERROR: 1>
    INFO: typing.ClassVar[LogLevel]  # value = <LogLevel.INFO: 0>
    WARNING: typing.ClassVar[LogLevel]  # value = <LogLevel.WARNING: 4>
    __members__: typing.ClassVar[dict[str, LogLevel]]  # value = {'INFO': <LogLevel.INFO: 0>, 'ERROR': <LogLevel.ERROR: 1>, 'DEBUG': <LogLevel.DEBUG: 2>, 'WARNING': <LogLevel.WARNING: 4>}
    @typing.overload
    def __eq__(self, other: LogLevel) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: LogLevel) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Model:
    """

    Simulation topology and physical parameters consumed by a Solver.

    All array properties are non-owning views of CUDA device memory. Their device
    allocations must remain valid while the solver uses this model.
    """
    def __init__(self) -> None:
        """
        Create an empty model with no referenced device data.
        """
    @property
    def affine_body_ids(self) -> Span:
        """
        Device span mapping affine bodies to Newton body indices.
        """
    @affine_body_ids.setter
    def affine_body_ids(self, arg1: Span) -> None:
        ...
    @property
    def affine_body_indices(self) -> Span:
        """
        Device span containing the four direct-ABD coordinate indices of each affine body.
        """
    @affine_body_indices.setter
    def affine_body_indices(self, arg1: Span) -> None:
        ...
    @property
    def body_centers_of_mass(self) -> Span:
        """
        Device span containing each affine body's body-frame center of mass.
        """
    @body_centers_of_mass.setter
    def body_centers_of_mass(self, arg1: Span) -> None:
        ...
    @property
    def cable_count(self) -> int:
        ...
    @cable_count.setter
    def cable_count(self, value: int) -> None:
        ...
    @property
    def cable_body_indices(self) -> Span:
        ...
    @cable_body_indices.setter
    def cable_body_indices(self, value: Span) -> None:
        ...
    @property
    def cable_parent_transforms(self) -> Span:
        ...
    @cable_parent_transforms.setter
    def cable_parent_transforms(self, value: Span) -> None:
        ...
    @property
    def cable_child_transforms(self) -> Span:
        ...
    @cable_child_transforms.setter
    def cable_child_transforms(self, value: Span) -> None:
        ...
    @property
    def cable_stiffnesses(self) -> Span:
        ...
    @cable_stiffnesses.setter
    def cable_stiffnesses(self, value: Span) -> None:
        ...
    @property
    def cable_dampings(self) -> Span:
        ...
    @cable_dampings.setter
    def cable_dampings(self, value: Span) -> None:
        ...
    @property
    def cable_rest_bend_twist(self) -> Span:
        ...
    @cable_rest_bend_twist.setter
    def cable_rest_bend_twist(self, value: Span) -> None:
        ...
    @property
    def cable_angular_bias(self) -> Span:
        ...
    @cable_angular_bias.setter
    def cable_angular_bias(self, value: Span) -> None:
        ...
    @property
    def cable_angular_hessian(self) -> Span:
        ...
    @cable_angular_hessian.setter
    def cable_angular_hessian(self, value: Span) -> None:
        ...
    @property
    def cable_enabled(self) -> Span:
        ...
    @cable_enabled.setter
    def cable_enabled(self, value: Span) -> None:
        ...
    @property
    def body_count(self) -> int:
        """
        Number of affine bodies in one static world template.
        """
    @body_count.setter
    def body_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def body_distance_constraint_count(self) -> int:
        """
        Number of augmented-Lagrangian hard distance constraints between affine bodies.
        """
    @body_distance_constraint_count.setter
    def body_distance_constraint_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def body_distance_constraint_indices(self) -> Span:
        """
        Device span containing the two affine-body indices of each hard distance constraint.
        """
    @body_distance_constraint_indices.setter
    def body_distance_constraint_indices(self, arg1: Span) -> None:
        ...
    @property
    def body_distance_constraint_mass_scales(self) -> Span:
        """
        Device span containing one mass-normalization scale per hard distance constraint.
        """
    @body_distance_constraint_mass_scales.setter
    def body_distance_constraint_mass_scales(self, arg1: Span) -> None:
        ...
    @property
    def body_distance_constraint_points0(self) -> Span:
        """
        Device span containing the first body-local point of each hard distance constraint.
        """
    @body_distance_constraint_points0.setter
    def body_distance_constraint_points0(self, arg1: Span) -> None:
        ...
    @property
    def body_distance_constraint_points1(self) -> Span:
        """
        Device span containing the second body-local point of each hard distance constraint.
        """
    @body_distance_constraint_points1.setter
    def body_distance_constraint_points1(self, arg1: Span) -> None:
        ...
    @property
    def body_distance_constraint_rest_lengths(self) -> Span:
        """
        Device span containing one rest length per hard distance constraint.
        """
    @body_distance_constraint_rest_lengths.setter
    def body_distance_constraint_rest_lengths(self, arg1: Span) -> None:
        ...
    @property
    def body_distance_constraint_weights0(self) -> Span:
        """
        Device span containing the first affine-body interpolation weights of each hard distance constraint.
        """
    @body_distance_constraint_weights0.setter
    def body_distance_constraint_weights0(self, arg1: Span) -> None:
        ...
    @property
    def body_distance_constraint_weights1(self) -> Span:
        """
        Device span containing the second affine-body interpolation weights of each hard distance constraint.
        """
    @body_distance_constraint_weights1.setter
    def body_distance_constraint_weights1(self, arg1: Span) -> None:
        ...
    @property
    def body_inertia_tensors(self) -> Span:
        """
        Device span containing one inertia tensor per affine body.
        """
    @body_inertia_tensors.setter
    def body_inertia_tensors(self, arg1: Span) -> None:
        ...
    @property
    def body_local_transforms(self) -> Span:
        """
        Device span containing one local transform per affine body.
        """
    @body_local_transforms.setter
    def body_local_transforms(self, arg1: Span) -> None:
        ...
    @property
    def body_masses(self) -> Span:
        """
        Device span containing one mass value per affine body.
        """
    @body_masses.setter
    def body_masses(self, arg1: Span) -> None:
        ...
    @property
    def body_point_line_count(self) -> int:
        """
        Number of point-to-line-segment constraints between affine bodies.
        """
    @body_point_line_count.setter
    def body_point_line_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def body_point_line_dampings(self) -> Span:
        """
        Device span containing one damping value per point-line constraint.
        """
    @body_point_line_dampings.setter
    def body_point_line_dampings(self, arg1: Span) -> None:
        ...
    @property
    def body_point_line_endpoints0(self) -> Span:
        """
        Device span containing the first body-local line endpoint of each point-line constraint.
        """
    @body_point_line_endpoints0.setter
    def body_point_line_endpoints0(self, arg1: Span) -> None:
        ...
    @property
    def body_point_line_endpoints1(self) -> Span:
        """
        Device span containing the second body-local line endpoint of each point-line constraint.
        """
    @body_point_line_endpoints1.setter
    def body_point_line_endpoints1(self, arg1: Span) -> None:
        ...
    @property
    def body_point_line_indices(self) -> Span:
        """
        Device span containing the point-body and line-body indices of each point-line constraint.
        """
    @body_point_line_indices.setter
    def body_point_line_indices(self, arg1: Span) -> None:
        ...
    @property
    def body_point_line_points(self) -> Span:
        """
        Device span containing the constrained body-local point of each point-line constraint.
        """
    @body_point_line_points.setter
    def body_point_line_points(self, arg1: Span) -> None:
        ...
    @property
    def body_point_line_stiffnesses(self) -> Span:
        """
        Device span containing one stiffness value per point-line constraint.
        """
    @body_point_line_stiffnesses.setter
    def body_point_line_stiffnesses(self, arg1: Span) -> None:
        ...
    @property
    def body_spring_count(self) -> int:
        """
        Number of point-to-point springs between affine bodies.
        """
    @body_spring_count.setter
    def body_spring_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def body_spring_indices(self) -> Span:
        """
        Device span containing the two affine-body indices of each body spring.
        """
    @body_spring_indices.setter
    def body_spring_indices(self, arg1: Span) -> None:
        ...
    @property
    def body_spring_points0(self) -> Span:
        """
        Device span containing the first body-local point of each body spring.
        """
    @body_spring_points0.setter
    def body_spring_points0(self, arg1: Span) -> None:
        ...
    @property
    def body_spring_points1(self) -> Span:
        """
        Device span containing the second body-local point of each body spring.
        """
    @body_spring_points1.setter
    def body_spring_points1(self, arg1: Span) -> None:
        ...
    @property
    def body_spring_rest_lengths(self) -> Span:
        """
        Device span containing one rest length per body spring.
        """
    @body_spring_rest_lengths.setter
    def body_spring_rest_lengths(self, arg1: Span) -> None:
        ...
    @property
    def body_spring_stiffnesses(self) -> Span:
        """
        Device span containing one stiffness value per body spring.
        """
    @body_spring_stiffnesses.setter
    def body_spring_stiffnesses(self, arg1: Span) -> None:
        ...
    @property
    def body_state_count(self) -> int:
        """
        Number of affine bodies in all dynamic world states.
        """
    @body_state_count.setter
    def body_state_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def collision_edge_count(self) -> int:
        """
        Number of collision surface edges in one static world template.
        """
    @collision_edge_count.setter
    def collision_edge_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def collision_edge_indices(self) -> Span:
        """
        Device span containing collision edge adjacency and endpoint particle indices.
        """
    @collision_edge_indices.setter
    def collision_edge_indices(self, arg1: Span) -> None:
        ...
    @property
    def collision_particle_count(self) -> int:
        """
        Number of active free particles in one static world template.
        """
    @collision_particle_count.setter
    def collision_particle_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def collision_particle_indices(self) -> Span:
        """
        Device span containing active unconstrained particle indices.
        """
    @collision_particle_indices.setter
    def collision_particle_indices(self, arg1: Span) -> None:
        ...
    @property
    def collision_triangle_count(self) -> int:
        """
        Number of collision surface triangles in one static world template.
        """
    @collision_triangle_count.setter
    def collision_triangle_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def collision_triangle_indices(self) -> Span:
        """
        Device span containing collision surface triangle particle indices.
        """
    @collision_triangle_indices.setter
    def collision_triangle_indices(self, arg1: Span) -> None:
        ...
    @property
    def edge_bending_cots(self) -> Span:
        """
        Device span containing four rest-pose cotangents per bending edge.
        """
    @edge_bending_cots.setter
    def edge_bending_cots(self, arg1: Span) -> None:
        ...
    @property
    def edge_bending_properties(self) -> Span:
        """
        Device span containing stiffness and damping for each bending edge.
        """
    @edge_bending_properties.setter
    def edge_bending_properties(self, arg1: Span) -> None:
        ...
    @property
    def edge_count(self) -> int:
        """
        Number of Style3D bending edges in the model.
        """
    @edge_count.setter
    def edge_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def edge_indices(self) -> Span:
        """
        Device span containing one four-particle bending stencil per edge.
        """
    @edge_indices.setter
    def edge_indices(self, arg1: Span) -> None:
        ...
    @property
    def edge_rest_areas(self) -> Span:
        """
        Device span containing the summed rest area adjacent to each bending edge.
        """
    @edge_rest_areas.setter
    def edge_rest_areas(self, arg1: Span) -> None:
        ...
    @property
    def fluid_boundary_count(self) -> int:
        ...
    @fluid_boundary_count.setter
    def fluid_boundary_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def fluid_boundary_positions(self) -> Span:
        """
        Fixed boundary quadrature positions; these are not collision shapes.
        """
    @fluid_boundary_positions.setter
    def fluid_boundary_positions(self, arg1: Span) -> None:
        ...
    @property
    def fluid_boundary_volumes(self) -> Span:
        """
        Positive boundary quadrature volumes in cubic meters.
        """
    @fluid_boundary_volumes.setter
    def fluid_boundary_volumes(self, arg1: Span) -> None:
        ...
    @property
    def fluid_bulk_modulus(self) -> float:
        ...
    @fluid_bulk_modulus.setter
    def fluid_bulk_modulus(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fluid_particle_count(self) -> int:
        ...
    @fluid_particle_count.setter
    def fluid_particle_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def fluid_particle_indices(self) -> Span:
        """
        Active fluid particle indices; distinct from elastic and affine topology.
        """
    @fluid_particle_indices.setter
    def fluid_particle_indices(self, arg1: Span) -> None:
        ...
    @property
    def fluid_rest_density(self) -> float:
        ...
    @fluid_rest_density.setter
    def fluid_rest_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fluid_smoothing_radius(self) -> float:
        ...
    @fluid_smoothing_radius.setter
    def fluid_smoothing_radius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fluid_viscosity(self) -> float:
        ...
    @fluid_viscosity.setter
    def fluid_viscosity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_adhesion(self) -> float:
        ...
    @granular_adhesion.setter
    def granular_adhesion(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_bond_count(self) -> int:
        ...
    @granular_bond_count.setter
    def granular_bond_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_bond_damping(self) -> float:
        ...
    @granular_bond_damping.setter
    def granular_bond_damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_bond_formation_gap(self) -> float:
        ...
    @granular_bond_formation_gap.setter
    def granular_bond_formation_gap(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_bond_formation_speed(self) -> float:
        ...
    @granular_bond_formation_speed.setter
    def granular_bond_formation_speed(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_bond_indices(self) -> Span:
        """
        Initial cohesive bond endpoint particle indices; fracture is irreversible within a solver.
        """
    @granular_bond_indices.setter
    def granular_bond_indices(self, arg1: Span) -> None:
        ...
    @property
    def granular_bond_rest_lengths(self) -> Span:
        """
        Initial bond rest lengths in meters; plastic history is owned by the solver.
        """
    @granular_bond_rest_lengths.setter
    def granular_bond_rest_lengths(self, arg1: Span) -> None:
        ...
    @property
    def granular_break_strain(self) -> float:
        ...
    @granular_break_strain.setter
    def granular_break_strain(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_cohesion(self) -> float:
        ...
    @granular_cohesion.setter
    def granular_cohesion(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_contact_stiffness(self) -> float:
        ...
    @granular_contact_stiffness.setter
    def granular_contact_stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_friction(self) -> float:
        ...
    @granular_friction.setter
    def granular_friction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_friction_damping(self) -> float:
        ...
    @granular_friction_damping.setter
    def granular_friction_damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_max_compaction(self) -> float:
        ...
    @granular_max_compaction.setter
    def granular_max_compaction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_max_dynamic_bonds(self) -> int:
        ...
    @granular_max_dynamic_bonds.setter
    def granular_max_dynamic_bonds(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_particle_count(self) -> int:
        ...
    @granular_particle_count.setter
    def granular_particle_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def granular_particle_indices(self) -> Span:
        """
        Granular particle indices; disjoint from fluid, elastic, and affine selections.
        """
    @granular_particle_indices.setter
    def granular_particle_indices(self, arg1: Span) -> None:
        ...
    @property
    def granular_yield_strain(self) -> float:
        ...
    @granular_yield_strain.setter
    def granular_yield_strain(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gravity(self) -> Span:
        """
        Device span containing one gravity vector per world.
        """
    @gravity.setter
    def gravity(self, arg1: Span) -> None:
        ...
    @property
    def hinge_count(self) -> int:
        """
        Number of bending hinges in the model.
        """
    @hinge_count.setter
    def hinge_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def hinge_indices(self) -> Span:
        """
        Device span containing the four particle indices of each hinge.
        """
    @hinge_indices.setter
    def hinge_indices(self, arg1: Span) -> None:
        ...
    @property
    def hinge_rest_angles(self) -> Span:
        """
        Device span containing one rest angle per hinge.
        """
    @hinge_rest_angles.setter
    def hinge_rest_angles(self, arg1: Span) -> None:
        ...
    @property
    def hinge_stiffnesses(self) -> Span:
        """
        Device span containing one bending stiffness value per hinge.
        """
    @hinge_stiffnesses.setter
    def hinge_stiffnesses(self, arg1: Span) -> None:
        ...
    @property
    def joint_body_point_line_count(self) -> int:
        """
        Number of joint-generated constraints at the end of the point-line arrays.
        """
    @joint_body_point_line_count.setter
    def joint_body_point_line_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def joint_body_spring_count(self) -> int:
        """
        Number of joint-generated springs at the end of the body spring arrays.
        """
    @joint_body_spring_count.setter
    def joint_body_spring_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def joint_constraint_mass_scale(self) -> float:
        """
        Mass-normalization scale shared by generated joint constraints.
        """
    @joint_constraint_mass_scale.setter
    def joint_constraint_mass_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def joint_dof_count(self) -> int:
        """
        Number of generalized joint-velocity coordinates.
        """
    @joint_dof_count.setter
    def joint_dof_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def joint_drive_body_indices(self) -> Span:
        """
        Device span containing each drive's parent and child affine-body indices.
        """
    @joint_drive_body_indices.setter
    def joint_drive_body_indices(self, arg1: Span) -> None:
        ...
    @property
    def joint_drive_child_origins(self) -> Span:
        """
        Device span containing child-frame joint origins.
        """
    @joint_drive_child_origins.setter
    def joint_drive_child_origins(self, arg1: Span) -> None:
        ...
    @property
    def joint_drive_child_reference_points(self) -> Span:
        """
        Device span containing child-frame angular reference points.
        """
    @joint_drive_child_reference_points.setter
    def joint_drive_child_reference_points(self, arg1: Span) -> None:
        ...
    @property
    def joint_drive_count(self) -> int:
        """
        Number of single-axis joint drives.
        """
    @joint_drive_count.setter
    def joint_drive_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def joint_drive_dof_indices(self) -> Span:
        """
        Device span mapping drives to generalized force and velocity coordinates.
        """
    @joint_drive_dof_indices.setter
    def joint_drive_dof_indices(self, arg1: Span) -> None:
        ...
    @property
    def joint_drive_parent_axis_points(self) -> Span:
        """
        Device span containing parent-frame unit-axis points.
        """
    @joint_drive_parent_axis_points.setter
    def joint_drive_parent_axis_points(self, arg1: Span) -> None:
        ...
    @property
    def joint_drive_parent_origins(self) -> Span:
        """
        Device span containing parent-frame joint origins.
        """
    @joint_drive_parent_origins.setter
    def joint_drive_parent_origins(self, arg1: Span) -> None:
        ...
    @property
    def joint_drive_parent_reference_points(self) -> Span:
        """
        Device span containing parent-frame angular reference points.
        """
    @joint_drive_parent_reference_points.setter
    def joint_drive_parent_reference_points(self, arg1: Span) -> None:
        ...
    @property
    def joint_drive_target_indices(self) -> Span:
        """
        Device span mapping drives to generalized position targets.
        """
    @joint_drive_target_indices.setter
    def joint_drive_target_indices(self, arg1: Span) -> None:
        ...
    @property
    def joint_drive_types(self) -> Span:
        """
        Device span containing each joint drive type.
        """
    @joint_drive_types.setter
    def joint_drive_types(self, arg1: Span) -> None:
        ...
    @property
    def joint_limit_lower(self) -> Span:
        """Optional per-DOF angular stop parameter."""
    @joint_limit_lower.setter
    def joint_limit_lower(self, arg1: Span) -> None:
        ...
    @property
    def joint_limit_upper(self) -> Span:
        """Optional per-DOF angular stop parameter."""
    @joint_limit_upper.setter
    def joint_limit_upper(self, arg1: Span) -> None:
        ...
    @property
    def joint_limit_stiffnesses(self) -> Span:
        """Optional per-DOF angular stop parameter."""
    @joint_limit_stiffnesses.setter
    def joint_limit_stiffnesses(self, arg1: Span) -> None:
        ...
    @property
    def joint_limit_dampings(self) -> Span:
        """Optional per-DOF angular stop parameter."""
    @joint_limit_dampings.setter
    def joint_limit_dampings(self, arg1: Span) -> None:
        ...
    @property
    def joint_effort_limits(self) -> Span:
        """
        Device span containing symmetric joint effort magnitude limits.
        """
    @joint_effort_limits.setter
    def joint_effort_limits(self, arg1: Span) -> None:
        ...
    @property
    def joint_target_count(self) -> int:
        """
        Number of generalized joint-position targets.
        """
    @joint_target_count.setter
    def joint_target_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def joint_target_dampings(self) -> Span:
        """
        Device span containing joint target damping values.
        """
    @joint_target_dampings.setter
    def joint_target_dampings(self, arg1: Span) -> None:
        ...
    @property
    def joint_target_modes(self) -> Span:
        """Optional Newton target mode per DOF (0 through 4)."""
    @joint_target_modes.setter
    def joint_target_modes(self, arg1: Span) -> None:
        ...
    @property
    def joint_passive_dampings(self) -> Span:
        """Optional passive viscous damping per DOF, independent of actuator limits."""
    @joint_passive_dampings.setter
    def joint_passive_dampings(self, arg1: Span) -> None:
        ...
    @property
    def joint_armature(self) -> Span:
        """Optional reflected inertia per scalar relative joint coordinate."""
    @joint_armature.setter
    def joint_armature(self, arg1: Span) -> None:
        ...
    @property
    def joint_target_stiffnesses(self) -> Span:
        """
        Device span containing joint target stiffness values.
        """
    @joint_target_stiffnesses.setter
    def joint_target_stiffnesses(self, arg1: Span) -> None:
        ...
    @property
    def particle_count(self) -> int:
        """
        Number of particles in one static world template.
        """
    @particle_count.setter
    def particle_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def particle_flags(self) -> Span:
        """
        Device span containing Newton particle flags.
        """
    @particle_flags.setter
    def particle_flags(self, arg1: Span) -> None:
        ...
    @property
    def particle_indices(self) -> Span:
        """
        Device span containing the index of each particle.
        """
    @particle_indices.setter
    def particle_indices(self, arg1: Span) -> None:
        ...
    @property
    def particle_mass_non_diagonal_count(self) -> int:
        """
        Number of symmetric off-diagonal particle mass entries.
        """
    @particle_mass_non_diagonal_count.setter
    def particle_mass_non_diagonal_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def particle_mass_non_diagonal_indices(self) -> Span:
        """
        Device span containing one particle-index pair per symmetric off-diagonal mass entry.
        """
    @particle_mass_non_diagonal_indices.setter
    def particle_mass_non_diagonal_indices(self, arg1: Span) -> None:
        ...
    @property
    def particle_mass_non_diagonal_values(self) -> Span:
        """
        Device span containing symmetric off-diagonal particle mass values.
        """
    @particle_mass_non_diagonal_values.setter
    def particle_mass_non_diagonal_values(self, arg1: Span) -> None:
        ...
    @property
    def particle_masses(self) -> Span:
        """
        Device span containing diagonal entries of the particle mass matrix.
        """
    @particle_masses.setter
    def particle_masses(self, arg1: Span) -> None:
        ...
    @property
    def particle_radii(self) -> Span:
        """
        Device span containing particle collision radii.
        """
    @particle_radii.setter
    def particle_radii(self, arg1: Span) -> None:
        ...
    @property
    def particle_rest_positions(self) -> Span:
        """
        Device span containing initial particle positions for all dynamic world states.
        """
    @particle_rest_positions.setter
    def particle_rest_positions(self, arg1: Span) -> None:
        ...
    @property
    def particle_state_count(self) -> int:
        """
        Number of particles in all dynamic world states.
        """
    @particle_state_count.setter
    def particle_state_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def particle_world(self) -> Span:
        """
        Device span containing the world index of each particle.
        """
    @particle_world.setter
    def particle_world(self, arg1: Span) -> None:
        ...
    @property
    def shape_bodies(self) -> Span:
        """
        Device span mapping collision shapes to Newton body indices.
        """
    @shape_bodies.setter
    def shape_bodies(self, arg1: Span) -> None:
        ...
    @property
    def shape_contact_dampings(self) -> Span:
        """
        Device span containing shape contact damping values.
        """
    @shape_contact_dampings.setter
    def shape_contact_dampings(self, arg1: Span) -> None:
        ...
    @property
    def shape_contact_friction_gains(self) -> Span:
        """
        Device span containing tangential friction gains [N s/m].
        """
    @shape_contact_friction_gains.setter
    def shape_contact_friction_gains(self, arg1: Span) -> None:
        ...
    @property
    def shape_contact_stiffnesses(self) -> Span:
        """
        Device span containing shape contact stiffness values.
        """
    @shape_contact_stiffnesses.setter
    def shape_contact_stiffnesses(self, arg1: Span) -> None:
        ...
    @property
    def shape_count(self) -> int:
        """
        Number of collision shapes in the model.
        """
    @shape_count.setter
    def shape_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def shape_frictions(self) -> Span:
        """
        Device span containing shape Coulomb-friction coefficients.
        """
    @shape_frictions.setter
    def shape_frictions(self, arg1: Span) -> None:
        ...
    @property
    def shape_margins(self) -> Span:
        """
        Device span containing shape collision margins.
        """
    @shape_margins.setter
    def shape_margins(self, arg1: Span) -> None:
        ...
    @property
    def shape_replicated_count(self) -> int:
        """
        Number of collision shapes replicated in each world.
        """
    @shape_replicated_count.setter
    def shape_replicated_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def shape_rolling_frictions(self) -> Span:
        """
        Optional dimensionless rolling-resistance coefficients; an empty span disables rolling resistance.
        """
    @shape_rolling_frictions.setter
    def shape_rolling_frictions(self, arg1: Span) -> None:
        ...
    @property
    def shape_scales(self) -> Span:
        """
        Device span containing collision geometry scales.
        """
    @shape_scales.setter
    def shape_scales(self, arg1: Span) -> None:
        ...
    @property
    def shape_torsional_frictions(self) -> Span:
        """
        Optional dimensionless contact spin-resistance coefficients; an empty span disables torsional friction.
        """
    @shape_torsional_frictions.setter
    def shape_torsional_frictions(self, arg1: Span) -> None:
        ...
    @property
    def shape_transforms(self) -> Span:
        """
        Device span containing body-local or world collision shape transforms.
        """
    @shape_transforms.setter
    def shape_transforms(self, arg1: Span) -> None:
        ...
    @property
    def shape_types(self) -> Span:
        """
        Device span containing Newton collision geometry types.
        """
    @shape_types.setter
    def shape_types(self, arg1: Span) -> None:
        ...
    @property
    def soft_contact_damping(self) -> float:
        """
        Particle contact damping.
        """
    @soft_contact_damping.setter
    def soft_contact_damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def soft_contact_friction(self) -> float:
        """
        Particle contact Coulomb-friction coefficient.
        """
    @soft_contact_friction.setter
    def soft_contact_friction(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def soft_contact_friction_gain(self) -> float:
        """
        Particle tangential friction gain [N s/m].
        """
    @soft_contact_friction_gain.setter
    def soft_contact_friction_gain(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def soft_contact_stiffness(self) -> float:
        """
        Particle contact stiffness.
        """
    @soft_contact_stiffness.setter
    def soft_contact_stiffness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def spring_count(self) -> int:
        """
        Number of distance springs in the model.
        """
    @spring_count.setter
    def spring_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def spring_indices(self) -> Span:
        """
        Device span containing the two particle indices of each spring.
        """
    @spring_indices.setter
    def spring_indices(self, arg1: Span) -> None:
        ...
    @property
    def spring_rest_lengths(self) -> Span:
        """
        Device span containing one rest length per spring.
        """
    @spring_rest_lengths.setter
    def spring_rest_lengths(self, arg1: Span) -> None:
        ...
    @property
    def spring_stiffnesses(self) -> Span:
        """
        Device span containing one stiffness value per spring.
        """
    @spring_stiffnesses.setter
    def spring_stiffnesses(self, arg1: Span) -> None:
        ...
    @property
    def tetrahedron_count(self) -> int:
        """
        Number of volumetric tetrahedra in one static world template.
        """
    @tetrahedron_count.setter
    def tetrahedron_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def tetrahedron_indices(self) -> Span:
        """
        Device span containing the four particle indices of each tetrahedron.
        """
    @tetrahedron_indices.setter
    def tetrahedron_indices(self, arg1: Span) -> None:
        ...
    @property
    def tetrahedron_inverse_dms(self) -> Span:
        """
        Device span containing one inverse rest-shape matrix per tetrahedron.
        """
    @tetrahedron_inverse_dms.setter
    def tetrahedron_inverse_dms(self, arg1: Span) -> None:
        ...
    @property
    def tetrahedron_materials(self) -> Span:
        """
        Device span containing k_mu, k_lambda, and k_damp per tetrahedron.
        """
    @tetrahedron_materials.setter
    def tetrahedron_materials(self, arg1: Span) -> None:
        ...
    @property
    def triangle_aniso_stiffnesses(self) -> Span:
        """
        Device span containing one Style3D anisotropic stiffness vector per triangle.
        """
    @triangle_aniso_stiffnesses.setter
    def triangle_aniso_stiffnesses(self, arg1: Span) -> None:
        ...
    @property
    def triangle_count(self) -> int:
        """
        Number of surface triangles in the model.
        """
    @triangle_count.setter
    def triangle_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def triangle_drag_coefficients(self) -> Span:
        """
        Optional nonnegative normal-drag Cd per collision triangle, or elastic triangle when collision topology is absent.
        """
    @triangle_drag_coefficients.setter
    def triangle_drag_coefficients(self, arg1: Span) -> None:
        ...
    @property
    def triangle_indices(self) -> Span:
        """
        Device span containing the three particle indices of each triangle.
        """
    @triangle_indices.setter
    def triangle_indices(self, arg1: Span) -> None:
        ...
    @property
    def triangle_inv_dms(self) -> Span:
        """
        Device span containing one inverse rest-shape matrix per triangle.
        """
    @triangle_inv_dms.setter
    def triangle_inv_dms(self, arg1: Span) -> None:
        ...
    @property
    def triangle_masses(self) -> Span:
        """
        Device span containing one mass value per triangle.
        """
    @triangle_masses.setter
    def triangle_masses(self, arg1: Span) -> None:
        ...
    @property
    def triangle_rest_areas(self) -> Span:
        """
        Device span containing one rest-area value per triangle.
        """
    @triangle_rest_areas.setter
    def triangle_rest_areas(self, arg1: Span) -> None:
        ...
    @property
    def world_count(self) -> int:
        """
        Number of simulation worlds in the model.
        """
    @world_count.setter
    def world_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class NonlinearSolver:
    """
    Members:

      NEWTON

      PD_NEWTON
    """
    NEWTON: typing.ClassVar[NonlinearSolver]  # value = <NonlinearSolver.NEWTON: 0>
    PD_NEWTON: typing.ClassVar[NonlinearSolver]  # value = <NonlinearSolver.PD_NEWTON: 1>
    __members__: typing.ClassVar[dict[str, NonlinearSolver]]  # value = {'NEWTON': <NonlinearSolver.NEWTON: 0>, 'PD_NEWTON': <NonlinearSolver.PD_NEWTON: 1>}
    @typing.overload
    def __eq__(self, other: NonlinearSolver) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: NonlinearSolver) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Solver:
    """

    Time-step solver for a Style3D Sim simulation model.

    Call initialize() once with a configured Model before calling step().
    """
    def __init__(self) -> None:
        """
        Create the default solver implementation.
        """
    def initialize(self, model: Model) -> None:
        """
        Initialize the solver for a model.

        The model's device allocations must remain valid while the solver uses them.
        Call this method before the first call to step().
        """
    def reserve_rigid_contact_capacity(self, capacity: typing.SupportsInt | typing.SupportsIndex) -> None:
        """Reserve contact state before CUDA graph capture."""
    def reset_joint_drive_state(self, stream: typing.Any = None, mask: Span = ...) -> None:
        """
        Reset persistent joint-coordinate branches after an external state reset.
        """
    def reset_cable_penalty_state(self, stream: typing.Any = None, mask: Span = ...) -> None:
        """Restore cable penalty seeds; an optional int mask selects flattened cable state rows."""
    def copy_cable_penalties(self, output: Span, stream: typing.Any = None) -> None:
        """Copy four current penalties per flattened cable state row into a device Vec4f span."""
    @typing.overload
    def step(self, state_in: State, state_out: State, dt: typing.SupportsFloat | typing.SupportsIndex, stream: typing.Any = None) -> None:
        """
        Advance without collision contacts or joint controls.
        """
    @typing.overload
    def step(self, state_in: State, state_out: State, contact: Contact, dt: typing.SupportsFloat | typing.SupportsIndex, stream: typing.Any = None) -> None:
        """
        Advance with collision contacts but without joint controls.
        """
    @typing.overload
    def step(self, state_in: State, state_out: State, control: Control, contact: Contact, dt: typing.SupportsFloat | typing.SupportsIndex, stream: typing.Any = None) -> None:
        """
        Advance the simulation by one time step.

        Parameters:
            state_in: State containing the input particle data.
            state_out: State whose referenced device memory receives the output data.
            control: Non-owning views of Newton's generalized joint-control buffers.
            contact: Non-owning views of the active Newton rigid-contact buffers.
            dt: Time-step duration.
            stream: CUDA Stream Protocol object, integer CUDA stream handle, or None for
                the default stream. All asynchronous work is submitted to this stream.

        initialize() must be called before this method. All referenced device
        allocations must remain valid until the solver work finishes.
        """
    @property
    def config(self) -> Config:
        """
        Mutable runtime solver configuration. Captured CUDA graphs must be rebuilt after changes.
        """
    @config.setter
    def config(self, arg0: Config) -> None:
        ...
class Span:
    """

    A non-owning view of a contiguous CUDA device-memory range.

    Span stores an opaque device address and an element count. It does not allocate,
    free, or dereference the referenced memory. Construction and field assignment
    reject a null address paired with a nonzero element count.
    """
    @typing.overload
    def __init__(self) -> None:
        """
        Create an empty span with a null address and zero elements.
        """
    @typing.overload
    def __init__(self, ptr: typing.SupportsInt | typing.SupportsIndex, size: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Create a non-owning span from a device address and an element count.
        """
    def __len__(self) -> int:
        """
        Return the number of elements in the span.
        """
    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the span.
        """
    @property
    def ptr(self) -> int:
        """
        Integer address of the first element in CUDA device memory.
        """
    @property
    def size(self) -> int:
        """
        Number of elements in the range, not its size in bytes.
        """
class State:
    """

    Particle positions and velocities used as solver input or output.

    Both properties are non-owning views of mutable CUDA device memory. Their
    device allocations must remain valid until the associated solver work finishes.
    """
    def __init__(self) -> None:
        """
        Create an empty state with no referenced device data.
        """
    @property
    def particle_f(self) -> Span:
        """
        Device span containing one external-force vector per particle.
        """
    @particle_f.setter
    def particle_f(self, arg1: Span) -> None:
        ...
    @property
    def particle_q(self) -> Span:
        """
        Device span containing one position vector per particle.
        """
    @particle_q.setter
    def particle_q(self, arg1: Span) -> None:
        ...
    @property
    def particle_qd(self) -> Span:
        """
        Device span containing one velocity vector per particle.
        """
    @particle_qd.setter
    def particle_qd(self, arg1: Span) -> None:
        ...
def is_logged_in() -> bool:
    """
    Return whether the session is authenticated.
    """
def login(name: str, password: str, force_signin: bool) -> bool:
    """
    Authenticate the Style3D Sim client.
    """
def set_log_callback(callback: collections.abc.Callable[[str, str, int, LogLevel, str], None]) -> None:
    """
    Register a callback invoked for each native log message.
    """
