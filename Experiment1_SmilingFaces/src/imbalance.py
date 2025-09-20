from datasets import load_dataset, load_from_disk
import pandas as pd
import numpy as np

# Load the dataset
dataset = load_from_disk("./CelebA-attrs")
3
# Define columns to exclude
exclude_columns = ['image', 'mask', 'image_and_mask','High_Cheekbones','Oval_Face','Rosy_Cheeks','Arched_Eyebrows','Bags_Under_Eyes','no_wgt']
exclude_prefixes = ['ipw_', 'aipw_', 'gcomp_']

# Get all column names
all_columns = dataset.column_names

# Filter out excluded columns
columns_to_keep = []
for col in all_columns:
    # Skip if it's in the explicit exclude list
    if col in exclude_columns:
        continue
    # Skip if it starts with any of the excluded prefixes
    if any(col.startswith(prefix) for prefix in exclude_prefixes):
        continue
    columns_to_keep.append(col)

# Remove unwanted columns from the dataset
dataset = dataset.select_columns(columns_to_keep)

# Now convert to pandas DataFrame (much smaller now)
df = dataset.to_pandas()

# Transform [-1,1] variables to [0,1]
# First, identify which columns have values in [-1,1] range
binary_columns = []
for col in df.columns:
    unique_vals = df[col].unique()
    if len(unique_vals) == 2 and set(unique_vals) == {-1, 1}:
        binary_columns.append(col)
        # Transform -1,1 to 0,1
        df[col] = (df[col] + 1) / 2

print(f"Transformed {len(binary_columns)} binary columns from [-1,1] to [0,1]")
print(f"Binary columns: {binary_columns}")

# Get columns to compare (everything except Smiling)
columns_to_compare = [col for col in df.columns if col != 'Smiling']

# Split data based on Smiling column (now 0 and 1)
smiling_group = df[df['Smiling'] == 1][columns_to_compare]
not_smiling_group = df[df['Smiling'] == 0][columns_to_compare]

# Calculate means for each group
smiling_means = smiling_group.mean()
not_smiling_means = not_smiling_group.mean()

# Calculate overall means (across all samples)
overall_means = df[columns_to_compare].mean()

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Overall_Mean': overall_means,
    'Smiling_Mean': smiling_means,
    'Not_Smiling_Mean': not_smiling_means,
    'Difference': smiling_means - not_smiling_means
})

# Sort by absolute difference to see the largest differences first
comparison_df['Abs_Difference'] = abs(comparison_df['Difference'])
comparison_df = comparison_df.sort_values('Abs_Difference', ascending=False)

print("\nMean comparison between Smiling==1 and Smiling==0:")
print("(Values are now proportions from 0 to 1)")
print(comparison_df.drop('Abs_Difference', axis=1))

# Optional: Show only the top 10 largest differences
print("\nTop 10 largest differences:")
print(comparison_df.head(10).drop('Abs_Difference', axis=1))