import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load JSON data
json_file_path = "/workspace/vlm/lmms-eval-updated_new/logging/logit_pope_gqa_pop_2.5_2.5_2.5_bilinear_s1.json"  # Replace with your JSON file path
output_dir = "/workspace/vlm/lmms-eval-updated_new/logging/stat_figs"  # Output directory for saving the plot
output_file = f"{output_dir}/pope_gqa_performance_3d_plot_with_baseline.png"

with open(json_file_path, "r") as file:
    data = json.load(file)

# Extract values from JSON
alphas = [entry["alpha"] for entry in data["all_results"]]
betas = [entry["beta"] for entry in data["all_results"]]
gammas = [entry["gamma"] for entry in data["all_results"]]
scores = [entry["score"] for entry in data["all_results"]]

# Convert to numpy arrays for reshaping (if necessary for 3D plots)
alphas = np.array(alphas)
betas = np.array(betas)
gammas = np.array(gammas)
scores = np.array(scores)

# Set the baseline score
baseline_score = 78.0

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
sc = ax.scatter(alphas, betas, gammas, c=scores, cmap="viridis", s=50)

# Labels and titles
ax.set_xlabel("Alpha", fontsize=12)
ax.set_ylabel("Beta", fontsize=12)
ax.set_zlabel("Gamma", fontsize=12)
ax.set_title("Performance by Alpha, Beta, Gamma with Baseline", fontsize=15)

# Add a color bar to represent scores
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label("Score", fontsize=12)

# Add a baseline line to the color bar
cbar.ax.axhline(y=baseline_score, color="red", linewidth=2, label=f"Baseline ({baseline_score:.3f})")
cbar.ax.legend(loc='upper left', fontsize=10)

# Save plot to the specified directory
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"3D plot with baseline saved to: {output_file}")

# Show plot (optional)
plt.show()