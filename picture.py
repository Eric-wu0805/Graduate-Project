import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman']

# The data is already in a usable dictionary format.
data = {
    'labels': [
        'Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS',
        'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms'
    ],
    'values': [
        37000, 18871, 11132, 6062, 4089,
        3496, 677, 583, 378, 44
    ]
}

# Create a DataFrame directly from the dictionary.
df = pd.DataFrame(data)

# Sort the DataFrame by values in descending order.
df_sorted = df.sort_values(by='values', ascending=False)

# Create a bar chart.
plt.figure(figsize=(15, 8))
plt.bar(df_sorted['labels'], df_sorted['values'])
plt.title('Distribution of Labels')
plt.xlabel('Labels')
plt.ylabel('Values')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('labels_distribution_NB15_bar_chart_times_new_roman.png')
print('Bar chart with Times New Roman font saved as labels_distribution_NB15_bar_chart_times_new_roman.png')