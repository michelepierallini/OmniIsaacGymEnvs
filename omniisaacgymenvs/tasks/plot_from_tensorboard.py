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

def plot_tensorboard_data(logdirs, tags, output_folder, 
                          font_size=24, 
                          label_width=5,
                          type_format='svg',
                          want_smooth=False,
                          want_filter=False,
                          smoothing_weight=0.6,
                          threshold=3,
                          log_names=None,
                          size_1=9, size_2=7):
    """Extracts data from TensorBoard logs and plots it using the specified style.

    Args:
        logdirs (list): List of paths to TensorBoard log directories.
        tags (list): List of tags (e.g., ['loss', 'accuracy']) to extract and plot.
        output_folder (str): Folder to save the plots.
        log_names (list): Names for each logdir to use in the legend.
    """

    # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # rc('text', usetex=True)
    # plt.rcParams['text.usetex'] = True

    # Handle log names
    if log_names is None:
        log_names = [f'Log {i+1}' for i in range(len(logdirs))]
    else:
        assert len(log_names) == len(logdirs), "log_names must have same length as logdirs"

    os.makedirs(output_folder, exist_ok=True)

    for tag in tags:
        plt.figure(figsize=(size_1, size_2))
        
        for logdir, log_label in zip(logdirs, log_names):
            # Load event data for this logdir
            event_acc = EventAccumulator(logdir)
            event_acc.Reload()
            
            # Extract scalar data
            try:
                scalar_events = event_acc.Scalars(tag)
            except KeyError:
                print(f"Tag {tag} not found in {logdir}. Skipping.")
                continue
                
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            
            # Process data
            if want_filter:
                values = filter_outliers(values, threshold=threshold)
            if want_smooth:
                values = smooth_data(values, smoothing_weight=smoothing_weight)
            
            # Plot data
            plt.plot(steps, values, linewidth=label_width, linestyle='-', label=log_label)

        # Configure plot
        plt.xlabel(r'Time\,\, [s]', fontsize=font_size)
        plt.ylabel(r'' + tag.replace('_', r'\,\,') + r'', fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.grid()
        plt.tick_params(labelsize=font_size)
        plt.tight_layout()

        # Save plot
        tag_basename = tag.split('/')[-1]  # Get last part of tag path
        output_path = os.path.join(output_folder, f'{tag_basename}.{type_format}')
        plt.savefig(output_path, format=type_format)
        plt.close()
        print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    logdirs = [
        "../runs/FishingRodPos_X_009_pos_new_2_Kpiu/summaries/",
        "../runs/FishingRodPos_X_009_pos_new_2_Kpiu_MB/summaries/",
        "../runs/FishingRodPos_X_009_pos_new_2_Kpiu_MB_pos/summaries/"
    ]
    tags = ["Episode/rew_err_pos", "Episode/rew_err_vel", "rewards/iter"]
    log_names = ["Kpiu", "MB", "MB_pos"]  # Custom names for each logdir
    output_folder = "comparison_plots"

    plot_tensorboard_data(
        logdirs=logdirs,
        tags=tags,
        output_folder=output_folder,
        log_names=log_names,
        want_smooth=True,
        smoothing_weight=0.6
    )