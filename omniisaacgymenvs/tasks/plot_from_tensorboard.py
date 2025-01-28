import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def filter_outliers(values, threshold=3):
    """Filter outliers using Z-score."""
    mean = np.mean(values)
    std = np.std(values)
    z_scores = [(value - mean) / std for value in values]
    return [value for value, z in zip(values, z_scores) if abs(z) <= threshold]

def smooth_data(values, smoothing_weight=0.6):
    """Apply exponential smoothing to the data."""
    smoothed_values = []
    last_value = values[0]
    for value in values:
        last_value = smoothing_weight * last_value + (1 - smoothing_weight) * value
        smoothed_values.append(last_value)
    return smoothed_values

def plot_tensorboard_data(logdir, tags, output_folder, 
                          font_size=24, 
                          label_width=5,
                          type_format='svg',
                          want_smooth=False,
                          smoothing_weight=0.6,
                          size_1=9, size_2=7):
    """Extracts data from TensorBoard logs and plots it using the specified style.

    Args:
        logdir (str): Path to the TensorBoard log directory.
        tags (list): List of tags (e.g., ['loss', 'accuracy']) to extract and plot.
        output_folder (str): Folder to save the plots.
    """
    # Set up matplotlib style
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rcParams['text.usetex'] = True

    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    os.makedirs(output_folder, exist_ok=True)

    for tag in tags:
        # Extract scalar data
        scalar_events = event_acc.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        
        if want_smooth:
            values = smooth_data(values, smoothing_weight)

        # Create plot
        plt.figure(figsize=(size_1, size_2))
        plt.plot(steps, values, linewidth=label_width, color='b', linestyle='solid', label=tag)
        plt.xlabel(r'$\mathbf{Time\,\, [s]}$', fontsize=font_size)
        plt.ylabel(r'$\mathbf{' + tag.replace('_', r'\,\,') + r'}$', fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.grid()
        plt.tick_params(labelsize=font_size)
        plt.tight_layout()

        # Save plot
        output_path = os.path.join(output_folder, f'{tag}.{type_format}')
        plt.savefig(output_path, format=type_format)
        plt.close()

        print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    logdir = "runs/FishingRodPos_X_009_pos_new_2_Kpiu_MB_pos/summaries/"  # log directory
    tags = ["Episode/rew_err_pos", "Episode/rew_err_vel", "rewards/iter"]
    output_folder = "output_plots"  

    plot_tensorboard_data(logdir, tags, output_folder)