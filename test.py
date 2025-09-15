import os
import pandas as pd

# Change this to your actual relative path
image_dir = 'images'  # No leading slash, relative path

# Step 1: Load full metadata and test list
all_df = pd.read_csv('Data_Entry_2017_v2020.csv')

with open('test_list.txt', 'r') as f:
    test_images = [line.strip() for line in f.readlines()]

# Step 2: Filter to test set
test_df = all_df[all_df['Image Index'].isin(test_images)].copy()
test_df['Pneumothorax'] = test_df['Finding Labels'].apply(
    lambda x: 1 if 'Pneumothorax' in x else 0
)
test_df = test_df[['Image Index', 'Pneumothorax']]

# Step 3: Check for exact image existence
valid_rows = []
missing_images = []

for idx, row in test_df.iterrows():
    img_path = os.path.join(image_dir, row['Image Index'])
    if os.path.isfile(img_path):
        valid_rows.append(row)
    else:
        missing_images.append(row['Image Index'])

print(f"⚠️  Missing images ({len(missing_images)}):")
for img in missing_images:
    print(f"    {img}")

# Create cleaned DataFrame
filtered_df = pd.DataFrame(valid_rows).reset_index(drop=True)

# Step 4: Save cleaned test.csv
filtered_df.to_csv('cleaned_test.csv', index=False)
print(f"\n✅ Cleaned test.csv saved with {len(filtered_df)} valid entries out of {len(test_df)}.")
