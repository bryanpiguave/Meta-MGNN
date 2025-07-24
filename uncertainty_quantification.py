import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

class UncertaintyQuantification:
    def __init__(self, result_df: pd.DataFrame, pred_column: str, label_column: str):
        # Input Validation: Ensure pred_column values are probabilities between 0 and 1
        if not (result_df[pred_column].between(0, 1).all()):
            raise ValueError(f"Predicted probabilities in '{pred_column}' must be between 0 and 1.")
        # Input Validation: Ensure label_column values are binary (0 or 1)
        if not (result_df[label_column].isin([0, 1]).all()):
            raise ValueError(f"True labels in '{label_column}' must be binary (0 or 1).")

        self.result_df = result_df
        self.pred_column = pred_column
        self.label_column = label_column

    def calculate_brier_score(self):
        # Vectorized calculation for efficiency
        brier_score = np.mean((self.result_df[self.pred_column] - self.result_df[self.label_column])**2)
        return brier_score

    def calculate_expected_calibration_error(self, n_bins: int = 10):
        preds = self.result_df[self.pred_column].values
        labels = self.result_df[self.label_column].values

        bins = np.linspace(0, 1, n_bins + 1)
        # Using np.digitize with `right=True` includes the rightmost bin edge,
        # which is common for the last bin (e.g., [0.9, 1.0]).
        # Then subtract 1 to get 0-indexed bin_indices.
        bin_indices = np.digitize(preds, bins, right=True) - 1
        
        # Adjust for values exactly 0.0, which might map to -1 with right=True on initial pass
        bin_indices[preds == 0.0] = 0 

        ece = 0.0
        total_samples = len(preds)

        for i in range(n_bins):
            # Mask for samples in the current bin.
            # For the last bin, include predictions == 1.0
            if i == n_bins - 1:
                bin_mask = (preds >= bins[i]) & (preds <= bins[i+1])
            else:
                bin_mask = (preds >= bins[i]) & (preds < bins[i+1])
            
            # Fallback to bin_indices if more robust for small edge cases
            # bin_mask = (bin_indices == i)

            bin_preds = preds[bin_mask]
            bin_labels = labels[bin_mask]

            num_in_bin = len(bin_preds)

            if num_in_bin > 0:
                avg_pred_prob = np.mean(bin_preds)
                bin_true_proportion = np.mean(bin_labels) # Mean of binary labels is the proportion of 1s
                
                bin_weight = num_in_bin / total_samples
                ece += bin_weight * np.abs(avg_pred_prob - bin_true_proportion)
        
        return ece

    def maximum_calibration_error(self, n_bins: int = 10):
        preds = self.result_df[self.pred_column].values
        labels = self.result_df[self.label_column].values

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(preds, bins, right=True) - 1
        bin_indices[preds == 0.0] = 0

        max_mce = 0.0
        
        for i in range(n_bins):
            if i == n_bins - 1:
                bin_mask = (preds >= bins[i]) & (preds <= bins[i+1])
            else:
                bin_mask = (preds >= bins[i]) & (preds < bins[i+1])
            
            # bin_mask = (bin_indices == i)

            bin_preds = preds[bin_mask]
            bin_labels = labels[bin_mask]

            num_in_bin = len(bin_preds)

            if num_in_bin > 0:
                avg_pred_prob = np.mean(bin_preds)
                bin_true_proportion = np.mean(bin_labels)
                
                max_mce = max(max_mce, np.abs(avg_pred_prob - bin_true_proportion))
        
        return max_mce

    def plot_reliability_diagram(self, n_bins: int = 10, title: str = "Reliability Diagram", save_path: str = None):
        """
        Generates and plots a reliability diagram for binary classification.

        Args:
            n_bins (int): The number of bins to divide the predicted probabilities into.
            title (str): The title of the plot.
            save_path (str, optional): Full path including filename to save the plot. If None, displays plot.
        """
        preds = self.result_df[self.pred_column].values
        labels = self.result_df[self.label_column].values

        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        # Arrays to store results for plotting
        avg_pred_probs_in_bin = []
        fraction_of_positives_in_bin = []
        bin_counts = []

        for i in range(n_bins):
            bin_lower_bound = bin_edges[i]
            bin_upper_bound = bin_edges[i+1]

            # Mask for samples within the current bin
            if i == n_bins - 1: # Last bin, include the upper bound (1.0)
                bin_mask = (preds >= bin_lower_bound) & (preds <= bin_upper_bound)
            else: # All other bins, exclude the upper bound
                bin_mask = (preds >= bin_lower_bound) & (preds < bin_upper_bound)

            bin_preds = preds[bin_mask]
            bin_labels = labels[bin_mask]
            
            num_in_bin = len(bin_preds)
            bin_counts.append(num_in_bin)

            if num_in_bin > 0:
                avg_pred_probs_in_bin.append(np.mean(bin_preds))
                fraction_of_positives_in_bin.append(np.mean(bin_labels))
            else:
                # If a bin is empty, we plot its midpoint but with NaN for the Y-value
                # so it's not part of the scatter plot line, but still shows for histogram.
                avg_pred_probs_in_bin.append(np.mean([bin_lower_bound, bin_upper_bound]))
                fraction_of_positives_in_bin.append(np.nan) # Mark as NaN

        # Filter out NaN values for the scatter plot (empty bins)
        plot_x = np.array(avg_pred_probs_in_bin)[~np.isnan(fraction_of_positives_in_bin)]
        plot_y = np.array(fraction_of_positives_in_bin)[~np.isnan(fraction_of_positives_in_bin)]
        
        # --- Plotting ---
        fig, ax1 = plt.subplots(figsize=(8, 8))

        # Plot the actual calibration points
        ax1.plot(plot_x, plot_y, 's-', label='Model Calibration', color='blue')
        
        # Plot the perfect calibration line
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

        # Add labels and title
        ax1.set_xlabel("Average Predicted Probability (Confidence)", fontsize=16)
        ax1.set_ylabel("Fraction of Positives (Accuracy)", fontsize=16)
        ax1.set_title(title, fontsize=16)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.legend(loc='upper left', fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_aspect('equal', adjustable='box') # Keep aspect ratio square

        # Optional: Add a second y-axis for the number of samples in each bin (histogram)
        ax2 = ax1.twinx()
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot a bar chart showing the count of samples in each bin
        ax2.bar(bin_midpoints, bin_counts, width=(bin_edges[1] - bin_edges[0]) * 0.8,
                alpha=0.2, color='gray', label='Number of Samples in Bin')
        ax2.set_ylabel("Number of Samples in Bin", fontsize=16)
        ax2.legend(loc='lower right', fontsize=16)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig) # Close the figure to free up memory
        else:
            plt.show()


def main():
    print("Calculating uncertainty quantification for task 0")
    result_df = pd.read_csv("probabilities_task_0.csv")
    uq = UncertaintyQuantification(result_df, "probability", "label")
    uq.plot_reliability_diagram(n_bins=10, title="Reliability Diagram Task 1", save_path="reliability_diagram_task_0.png")
    uq.calculate_expected_calibration_error(n_bins=10)
    uq.maximum_calibration_error(n_bins=10)

    print(f"Brier Score: {uq.calculate_brier_score()}")
    print(f"Expected Calibration Error: {uq.calculate_expected_calibration_error(n_bins=10)}")
    print(f"Maximum Calibration Error: {uq.maximum_calibration_error(n_bins=10)}")

    print("Calculating uncertainty quantification for task 1")
    result_df = pd.read_csv("probabilities_task_1.csv")
    uq = UncertaintyQuantification(result_df, "probability", "label")
    uq.plot_reliability_diagram(n_bins=10, title="Reliability Diagram Task 2", save_path="reliability_diagram_task_1.png")
    uq.calculate_expected_calibration_error(n_bins=10)
    uq.maximum_calibration_error(n_bins=10)
    print(f"Brier Score: {uq.calculate_brier_score()}")
    print(f"Expected Calibration Error: {uq.calculate_expected_calibration_error(n_bins=10)}")
    print(f"Maximum Calibration Error: {uq.maximum_calibration_error(n_bins=10)}")

    print("Calculating uncertainty quantification for task 2")
    result_df = pd.read_csv("probabilities_task_2.csv")
    uq = UncertaintyQuantification(result_df, "probability", "label")
    uq.plot_reliability_diagram(n_bins=10, title="Reliability Diagram Task 3", save_path="reliability_diagram_task_2.png")
    uq.calculate_expected_calibration_error(n_bins=10)
    uq.maximum_calibration_error(n_bins=10)
    print(f"Brier Score: {uq.calculate_brier_score()}")
    print(f"Expected Calibration Error: {uq.calculate_expected_calibration_error(n_bins=10)}")
    print(f"Maximum Calibration Error: {uq.maximum_calibration_error(n_bins=10)}")



if __name__ == "__main__":
    main()