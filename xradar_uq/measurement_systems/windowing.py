# RangeSensor r < rthresh
# AnglesOnly uses r_0 < r < r_1
# Radar uses r_0 < r < r_1

def tracking_measurability(state, predicted_state, elevation_fov=(5 * jnp.pi / 180), azimuth_fov=(5 * jnp.pi / 180), range_fov=1):

    rho = np.linalg.norm(state[:3])
    rho_pred = np.linalg.norm(predicted_state[:3])
    elevation = np.arcsin(state[2] / rho)
    elevation_pred = np.arcsin(predicted_state[2] / rho_pred)
    azimuth = np.arctan2(state[1], state[0])
    azimuth_pred = np.arctan2(predicted_state[1], predicted_state[0])
    
    return (np.abs(elevation - elevation_pred) <= elevation_fov / 2 and
            np.abs(azimuth - azimuth_pred) <= azimuth_fov / 2)
