
import matplotlib.pyplot as plt

def get_muscle_lengths_debug(state, muscle_attachment_points, body_indices, muscle_indices):
    pos = state[..., 0:3]
    rot = state[..., 3:7]

    pos_sel = pos[..., body_indices, :]
    rot_sel = rot[..., body_indices, :]

    offsets = muscle_attachment_points
    if offsets.shape[-1] != 3:
        offsets = offsets.T

    while offsets.dim() < pos_sel.dim():
        offsets = offsets.unsqueeze(0)

    offsets = offsets.expand_as(pos_sel)

    rotated = broadcast_quat_apply(rot_sel, offsets)
    waypoints = pos_sel + rotated

    return waypoints

def plot_muscles_3d(waypoints, muscle_indices):
    """
    waypoints: [N, 3] or [B, N, 3]
    muscle_indices: [N]
    """

    wp = waypoints.detach().cpu().numpy()
    mi = muscle_indices.detach().cpu().numpy()

    # handle batch (take first frame if batched)
    if wp.ndim == 3:
        wp = wp[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_points = wp.shape[0]

    # draw segments
    for i in range(num_points - 1):

        # only connect if same muscle chain
        if mi[i + 1] >= mi[i]:  # your reset logic
            ax.plot(
                [wp[i, 0], wp[i + 1, 0]],
                [wp[i, 1], wp[i + 1, 1]],
                [wp[i, 2], wp[i + 1, 2]],
                color='blue'
            )

    # scatter points
    ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c=mi, cmap='jet')

    ax.set_title("3D Muscle Waypoints")
    plt.show()