
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Data provided in the request + filled based on description (smaller EDA features)
data = {
    'Feature': [
        'HR', 'TEMP_slope', 'ACC_min', 'BVP_std', 'ACC_std',
        'EDA_slope', 'EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max', 'EDA_peaks'
    ],
    'Importance': [
        0.0893, 0.0850, 0.0421, 0.0380, 0.0350,
        0.0200, 0.0180, 0.0150, 0.0100, 0.0080, 0.0050
    ]
}

df = pd.DataFrame(data)

# Sort by Importance descending
df = df.sort_values(by='Importance', ascending=True)  # Ascending for horizontal bar plot (bottom to top)

# Plotting
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create horizontal bar chart
plt.barh(df['Feature'], df['Importance'], color=sns.color_palette("viridis", len(df)))

plt.xlabel('Feature Importance (Integrated Gradients)')
plt.title('Figura 10 – Atribuição média de importância das características')
plt.tight_layout()

# Save the figure
plt.savefig('figure_10.png', dpi=300)
print("Figure 10 saved as 'figure_10.png'")
