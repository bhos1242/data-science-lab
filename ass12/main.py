import numpy as np
import matplotlib.pyplot as plt

# Given vectors
colors = ["green", "orange", "brown"]
months = ["Mar", "Apr", "May", "Jun", "Jul"]
regions = ["East", "West", "North"]

# Example revenue data for each region by month
revenue = np.array([
  [120, 130, 140, 150, 160],  # East
  [100, 110, 105, 115, 120],  # West
  [80,  90,  95, 100, 105]    # North
])

# Create Stacked Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

# Create the stacked bar chart
x = np.arange(len(months))
width = 0.6

# Plot each region's data
bottom = np.zeros(len(months))
for i, (region, color) in enumerate(zip(regions, colors)):
    ax.bar(x, revenue[i], width, label=region, color=color, bottom=bottom)
    bottom += revenue[i]

# Customize the chart
ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Revenue', fontsize=12, fontweight='bold')
ax.set_title('Monthly Revenue by Region', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('revenue_stacked_bar_chart.png', dpi=300, bbox_inches='tight')
print("✅ Stacked bar chart created successfully!")
print("📊 Chart saved as: revenue_stacked_bar_chart.png")
plt.show()
