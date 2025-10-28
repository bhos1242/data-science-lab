import pandas as pd
import matplotlib.pyplot as plt

# Read the moviesData.csv file
df = pd.read_csv('moviesData.csv')

print("✅ Dataset loaded successfully!")
print(f"📊 Total movies: {len(df)}")
print("\nFirst 10 movies:")
print(df[['title', 'critics_score']].head(10))

# Get the first 10 movies
first_10_movies = df.head(10)

# Create bar chart of critics_score for the first 10 movies
fig, ax = plt.subplots(figsize=(14, 8))

# Create bar chart
bars = ax.bar(range(len(first_10_movies)), 
               first_10_movies['critics_score'], 
               color='steelblue', 
               edgecolor='black', 
               linewidth=1.5,
               alpha=0.7)

# Color bars based on score (green for high, yellow for medium, red for low)
colors = []
for score in first_10_movies['critics_score']:
    if score >= 90:
        colors.append('#2ecc71')  # Green
    elif score >= 80:
        colors.append('#f39c12')  # Orange
    else:
        colors.append('#e74c3c')  # Red

for bar, color in zip(bars, colors):
    bar.set_color(color)

# Customize the chart
ax.set_xlabel('Movie', fontsize=12, fontweight='bold')
ax.set_ylabel('Critics Score', fontsize=12, fontweight='bold')
ax.set_title('Critics Score for First 10 Movies', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(first_10_movies)))
ax.set_xticklabels(first_10_movies['title'], rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on top of bars
for i, (idx, row) in enumerate(first_10_movies.iterrows()):
    ax.text(i, row['critics_score'] + 1, str(row['critics_score']), 
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add a legend for color coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Excellent (90+)'),
    Patch(facecolor='#f39c12', label='Good (80-89)'),
    Patch(facecolor='#e74c3c', label='Fair (<80)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()

# Save the plot
plt.savefig('critics_score_bar_chart.png', dpi=300, bbox_inches='tight')
print("\n✅ Bar chart created successfully!")
print("💾 Chart saved as: critics_score_bar_chart.png")

# Show the plot
plt.show()

# Print statistics
print("\n📈 Statistics:")
print(f"Average Critics Score (first 10): {first_10_movies['critics_score'].mean():.2f}")
print(f"Highest Score: {first_10_movies['critics_score'].max()} ({first_10_movies.loc[first_10_movies['critics_score'].idxmax(), 'title']})")
print(f"Lowest Score: {first_10_movies['critics_score'].min()} ({first_10_movies.loc[first_10_movies['critics_score'].idxmin(), 'title']})")
