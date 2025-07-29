import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

delta_v_range = jnp.linspace(0.001, 0.1, 20)
maneuver_proportion_range = jnp.linspace(0.0, 1.0, 11)
measurement_range = jnp.linspace(0.0015968913847709792, 3.5, 1000)

measurement_frequency_results_engmf = jnp.load('cache/measurement_frequency/results_sweep_EnGMF.npy')
measurement_frequency_results_ukf = jnp.load('cache/measurement_frequency/results_sweep_UKF_1.npy')

def fix_zero_maneuver_consistency(data):
    """Ensure that when maneuver_proportion=0%, all delta_v values give identical results"""
    fixed_data = data.copy()
    zero_maneuver_baseline = data[:, 0, 0]  # Use first delta_v as baseline for 0% maneuver
    
    for delta_v_idx in range(data.shape[1]):
        fixed_data = fixed_data.at[:, delta_v_idx, 0].set(zero_maneuver_baseline)
    
    return fixed_data

measurement_frequency_results_engmf = fix_zero_maneuver_consistency(measurement_frequency_results_engmf)
measurement_frequency_results_ukf = fix_zero_maneuver_consistency(measurement_frequency_results_ukf)

measurement_frequency_results_engmf = measurement_frequency_results_engmf[:, ::2, ::2]
measurement_frequency_results_ukf = measurement_frequency_results_ukf[:, ::2, ::2]
delta_v_range = delta_v_range[::2]
maneuver_proportion_range = maneuver_proportion_range[::2]

def tu_to_human_time(tu_value):
    seconds_per_tu = 382981.0
    total_seconds = float(tu_value) * seconds_per_tu
    
    time_units = [
        (31536000, "years"), (86400, "days"), (3600, "hours"), 
        (60, "min."), (1, "seconds")
    ]
    
    for divisor, unit_name in time_units:
        unit_value = total_seconds / divisor
        if unit_value >= 1.0:
            return f"{unit_value:.2f} {unit_name}"
    
    return f"{total_seconds:.2e} seconds"

def find_custody_loss_index(found_proportions):
    threshold = 0.9
    consecutive_count = 3
    
    found_proportions_array = np.array(found_proportions)
    
    for i in range(len(found_proportions_array) - consecutive_count + 1):
        consecutive_below = np.all(found_proportions_array[i:i+consecutive_count] < threshold)
        if consecutive_below:
            return i
    
    return len(found_proportions_array) - 1

def compute_custody_matrix(measurement_data):
    custody_times_matrix = []
    custody_times_human = []
    
    for delta_v_idx in range(len(delta_v_range)):
        row_times = []
        row_human = []
        for maneuver_prop_idx in range(len(maneuver_proportion_range)):
            found_proportions = measurement_data[:, delta_v_idx, maneuver_prop_idx]
            custody_loss_idx = find_custody_loss_index(found_proportions)
            custody_time_tu = measurement_range[custody_loss_idx]
            custody_time_human = tu_to_human_time(custody_time_tu)
            
            row_times.append(float(custody_time_tu))
            row_human.append(custody_time_human)
        
        custody_times_matrix.append(row_times)
        custody_times_human.append(row_human)
    
    return np.array(custody_times_matrix), custody_times_human

du_tu_to_m_s_conversion = (389703 / 382981) * 1000

custody_matrix_engmf, custody_human_engmf = compute_custody_matrix(measurement_frequency_results_engmf)
custody_matrix_ukf, custody_human_ukf = compute_custody_matrix(measurement_frequency_results_ukf)

Path("figures/custody_maintenance").mkdir(parents=True, exist_ok=True)

delta_v_m_s = np.array(delta_v_range) * du_tu_to_m_s_conversion
maneuver_proportion_percent = np.array(maneuver_proportion_range) * 100

custody_matrix_engmf, custody_human_engmf = compute_custody_matrix(measurement_frequency_results_engmf)
custody_matrix_ukf, custody_human_ukf = compute_custody_matrix(measurement_frequency_results_ukf)

combined_min = min(np.min(custody_matrix_engmf), np.min(custody_matrix_ukf))
combined_max = max(np.max(custody_matrix_engmf), np.max(custody_matrix_ukf))

norm = None
cbar_label = 'Minimum Observation Frequency to Maintain Custody'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')
fig.patch.set_facecolor('black')

from matplotlib.colors import LogNorm

norm = LogNorm(vmin=combined_min, vmax=combined_max)

heatmap1 = sns.heatmap(custody_matrix_engmf, 
                      xticklabels=[f"{prop:.0f}%" for prop in maneuver_proportion_percent],
                      yticklabels=[f"{dv:.1f}" for dv in delta_v_m_s],
                      annot=np.array(custody_human_engmf), 
                      fmt='', 
                      cmap='plasma',
                      norm=norm,
                      vmin=combined_min,
                      vmax=combined_max,
                      cbar=False,
                      annot_kws={'fontsize': 14},
                      ax=ax1)

heatmap2 = sns.heatmap(custody_matrix_ukf, 
                      xticklabels=[f"{prop:.0f}%" for prop in maneuver_proportion_percent],
                      yticklabels=[f"{dv:.1f}" for dv in delta_v_m_s],
                      annot=np.array(custody_human_ukf), 
                      fmt='', 
                      cmap='plasma',
                      norm=norm,
                      vmin=combined_min,
                      vmax=combined_max,
                      cbar=False,
                      annot_kws={'fontsize': 14},
                      ax=ax2)

# # Create global horizontal colorbar
# cbar_ax = fig.add_axes([0.15, -0.1, 0.7, 0.03])  # [left, bottom, width, height]
# cbar = fig.colorbar(heatmap2.collections[0], cax=cbar_ax, orientation='horizontal')

# # Style the global colorbar
# cbar_ticks_tu = np.linspace(combined_min, combined_max, 8)

# cbar_labels = [tu_to_human_time(tick) for tick in cbar_ticks_tu]
# cbar.set_ticks(cbar_ticks_tu)
# cbar.set_ticklabels(cbar_labels, weight='bold')
# cbar.set_label(cbar_label, fontsize=12, weight='bold')
# cbar.ax.tick_params(colors='white')
# cbar.ax.xaxis.label.set_color('white')

##
print(f"Combined min: {combined_min}")
print(f"Combined max: {combined_max}")
print(f"EnGMF actual range: {np.min(custody_matrix_engmf)} to {np.max(custody_matrix_engmf)}")
print(f"UKF actual range: {np.min(custody_matrix_ukf)} to {np.max(custody_matrix_ukf)}")

# Create global horizontal colorbar
cbar_ax = fig.add_axes([0.15, -0.025, 0.8, 0.03])

pos1 = ax1.get_position()
pos2 = ax2.get_position()

left = pos1.x0
width = pos2.x1 - pos1.x0
bottom = -0.025
height = 0.03

cbar_ax = fig.add_axes([left, bottom, width, height])

cbar = fig.colorbar(heatmap2.collections[0], cax=cbar_ax, orientation='horizontal')

# Style the global colorbar with log-spaced ticks
n_ticks = 6
cbar_ticks_tu = np.logspace(np.log10(combined_min), np.log10(combined_max), n_ticks)

actual_min = min(np.min(custody_matrix_engmf), np.min(custody_matrix_ukf))
actual_max = max(np.max(custody_matrix_engmf), np.max(custody_matrix_ukf))

norm = LogNorm(vmin=actual_min, vmax=actual_max)

# Use actual range for colorbar ticks
cbar_ticks_tu = np.logspace(np.log10(actual_min), np.log10(actual_max), n_ticks)

cbar_labels = [tu_to_human_time(tick) for tick in cbar_ticks_tu]
cbar.set_ticks(cbar_ticks_tu)
cbar.set_ticklabels(cbar_labels, weight='bold')
cbar.set_label(cbar_label, fontsize=14, weight='bold')

# Set colorbar text to white
cbar.ax.tick_params(colors='white')
cbar.ax.yaxis.label.set_color('white')


##

# n_ticks = 6
# cbar_ticks_tu = np.logspace(combined_min, combined_max, n_ticks)

# cbar_labels = [tu_to_human_time(tick) for tick in cbar_ticks_tu]
# cbar.set_ticks(cbar_ticks_tu)
# cbar.set_ticklabels(cbar_labels, weight='bold')

# # After setting ticks and labels
# for i, label in enumerate(cbar.ax.get_xticklabels()):
#     if i % 2 == 1:  # Every other label
#         label.set_verticalalignment('top')
#         label.set_y(-0.5)  # Move below axis
#     else:
#         label.set_verticalalignment('bottom')
#         label.set_y(-2)   # Move above axis

ax2.set_yticklabels([])

ax1.tick_params(axis='y', rotation=0, labelsize=14)
ax1.tick_params(axis='x', rotation=0, labelsize=14)
ax2.tick_params(axis='x', rotation=0, labelsize=14)


# cbar = heatmap2.collections[0].colorbar
# cbar_ticks_tu = np.linspace(combined_min, combined_max, 8)
# cbar_labels = [tu_to_human_time(tick) for tick in cbar_ticks_tu]
# cbar.set_ticks(cbar_ticks_tu)
# cbar.set_ticklabels(cbar_labels, fontsize=14)
# cbar.set_label(cbar_label, fontsize=14)

ax1.set_xlabel('Maneuver Proportion (%)', fontsize=14)
ax1.set_ylabel('ΔV Magnitude (m/s)', fontsize=14)
ax1.set_title('EnGMF Filter', fontsize=18, pad=20)

ax2.set_xlabel('Maneuver Proportion (%)', fontsize=14)
ax2.set_title('UKF Filter', fontsize=18, pad=20)

# fig.suptitle('Minimum Measurement Cadence to Maintain Custody', 
#              fontsize=16, y=0.98)
# Add after creating the heatmaps, before plt.tight_layout()

# Set all text to white for both axes
for ax in [ax1, ax2]:
    ax.tick_params(colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    # # Set annotation text to white
    # for text in ax.texts:
    #     text.set_color('white')


# Set main title to white
# fig.suptitle('Minimum Measurement Cadence to Maintain Custody', 
#              fontsize=16, y=0.98, color='white')

plt.tight_layout()
plt.savefig('figures/custody_maintenance/custody_comparison_heatmap_black.png', dpi=300, bbox_inches='tight', facecolor='black')

assert custody_matrix_engmf.shape == (10, 6)
assert custody_matrix_ukf.shape == (10, 6)
assert len(delta_v_m_s) == 10
assert len(maneuver_proportion_percent) == 6

print(f"EnGMF matrix shape: {custody_matrix_engmf.shape}")
print(f"UKF matrix shape: {custody_matrix_ukf.shape}")
print(f"Comparison heatmap saved successfully")
