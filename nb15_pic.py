import pandas as pd
import matplotlib.pyplot as plt

# The data from the table in the image
data = {
    'Attack Types': [
        'Normal', 
        'Generic', 
        'Exploits', 
        'Fuzzers', 
        'DoS', 
        'Reconnaissance', 
        'Analysis', 
        'Backdoor', 
        'Shellcode', 
        'Worms'
    ],
    'Absolute Frequencies': [
        37000, 
        18871, 
        11132, 
        6062, 
        4089, 
        3496, 
        677, 
        583, 
        378, 
        44
    ]
}

# Create a pandas DataFrame from the data
df = pd.DataFrame(data)

# Sort the data by frequency in descending order for better visualization
df_sorted = df.sort_values(by='Absolute Frequencies', ascending=False)

# Create a bar chart
plt.figure(figsize=(12, 7))
plt.bar(df_sorted['Attack Types'], df_sorted['Absolute Frequencies'], color='skyblue')

# Add labels and a title
plt.xlabel('攻擊類型 (Attack Types)')
plt.ylabel('絕對頻率 (Absolute Frequencies)')
plt.title('不同攻擊類型的絕對頻率分佈')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent labels from being cut off
plt.tight_layout()

# Save the plot to a file
plt.savefig('attack_types_bar_chart.png')

# Display the plot
plt.show()

print("長條圖已生成並儲存為 'attack_types_bar_chart.png'。")