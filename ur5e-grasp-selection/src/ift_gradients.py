import numpy as np
from pydrake.all import (
    MultibodyPlant,
    JacobianWrtVariable,
)

def velocity_indices_for_model(plant, model_instance):
    """
    Velocity indices of a model instance's joints within the full plant.

    Fixed for the lifetime of a finalized plant, so callers cache it rather than walking
    every joint in the scene on each Jacobian evaluation.
    """
    indices = []
    for i in plant.GetJointIndices():
        joint = plant.get_joint(i)
        if joint.model_instance() == model_instance:
            for k in range(joint.num_velocities()):
                indices.append(joint.velocity_start() + k)
    return indices


def plant_jacobian_for_model(plant, context, model_instance, frame_E, frame_W,
                             indices=None):
    """
    Helper to get the 6xNV Jacobian for a specific model instance in a larger plant.
    """
    J_full = plant.CalcJacobianSpatialVelocity(
        context,
        JacobianWrtVariable.kV,
        frame_E,
        np.zeros(3),
        frame_W,
        frame_W
    )

    if indices is None:
        indices = velocity_indices_for_model(plant, model_instance)

    return J_full[:, indices]

def _damped_pinv(J, lam_eff):
    """
    The Tikhonov-damped pseudo-inverse V diag(s/(s^2+lam)) U^T.

    Computed through the SVD *deliberately*, even though (J^T J + lam I)^-1 J^T is
    algebraically identical and a 6x6 solve is cheaper than a 6x6 SVD.  Forming J^T J
    squares the condition number, and near-singular J is not an edge case here -- it is
    the regime the whole boundary formulation is about.  Measured on this problem, the
    normal-equations form disagrees with the SVD form by up to 2e-9 in dq/dp (against
    ~1e-15 for a well-conditioned J), which is enough to send SNOPT down a different
    trajectory and change ~18% of run outcomes.  The SVD is the accurate one, and the
    speed it costs is small next to the GetQ cache.

    lam_eff == 0 is the undamped case, where the damped form degenerates, so fall back to
    the plain pseudo-inverse -- which is what the limit means.
    """
    if lam_eff <= 0.0:
        return np.linalg.pinv(J)
    U, S, Vt = np.linalg.svd(J)
    return Vt.T @ np.diag(S / (S**2 + lam_eff)) @ U.T


class IftGradient:
    def __init__(self, plant: MultibodyPlant, arm_model_instance):
        self.plant = plant
        self.arm_model_instance = arm_model_instance
        self.context = self.plant.CreateDefaultContext()
        self.frame_E = self.plant.GetFrameByName("tool0", self.arm_model_instance)
        self.frame_W = self.plant.world_frame()
        self._v_indices = velocity_indices_for_model(self.plant, self.arm_model_instance)

    #: The damping strategies `compute_dq_dpose` accepts.  Exposed so argument parsers can
    #: reject a typo at parse time rather than mid-solve.
    STRATEGIES = ("residual", "lm_constant", "pinv")

    def compute_dq_dpose(self, q, residual_6d=None, lam=1e-4, strategy="residual"):
        """
        Compute dq/dpose = J_inv based on the specified gradient approximation strategy.
        `residual_6d` should be a 6D numpy array representing the spatial error.

        "residual" is the default and produces every reported number.  "lm_constant" and
        "pinv" are kept because both are cited as measured comparisons: the undamped
        "pinv" is what inverts the Direct formulation's advantage, which is the evidence
        that the damping is load-bearing rather than cosmetic.
        """
        self.plant.SetPositions(self.context, self.arm_model_instance, q)
        J = plant_jacobian_for_model(self.plant, self.context, self.arm_model_instance,
                                     self.frame_E, self.frame_W, indices=self._v_indices)

        if strategy == "pinv":
            return np.linalg.pinv(J)

        elif strategy == "lm_constant":
            # (J^T J + lam * I)^-1 J^T.  Left in the normal-equations form it has always
            # used, rather than routed through _damped_pinv, so its numerics are unchanged.
            return np.linalg.solve(J.T @ J + lam * np.eye(6), J.T)

        elif strategy == "residual":
            if residual_6d is None:
                raise ValueError("residual_6d must be provided for residual strategy")
            return _damped_pinv(J, lam * np.linalg.norm(residual_6d))

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Expected one of {self.STRATEGIES}.")
