"""Optional measured-layout experiments; the canonical scene is unchanged."""
import numpy as np


def front_mount(model):
    """Put the insertion wrist's pitch axis along motor roll; initialization only."""
    b=model.body('right_base').id
    model.body_pos[b]=[0.,-.20,.0024]
    model.body_quat[b]=[np.sqrt(.5),0.,0.,np.sqrt(.5)]
    # Keep loose pickup inside the same arm's vertical-grasp workspace.
    q=model.jnt_qposadr[model.joint('motor_free').id]
    model.qpos0[q:q+2]=[.10,.02]
    model.body_pos[model.body('motor').id,:2]=[.10,.02]


def opposed_mounts(model):
    """Horizontal rear base grasp and vertical front motor insertion."""
    front_mount(model)
    b=model.body('left_base').id
    model.body_pos[b]=[-.011,.40,.0024]
    model.body_quat[b]=[np.sqrt(.5),0.,0.,-np.sqrt(.5)]
