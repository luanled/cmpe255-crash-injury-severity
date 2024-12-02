import pandas as pd
from sklearn.model_selection import train_test_split

# Load the preprocessed datasets
crashes_data_path = '../data/processed/preprocessed_crashes_data.csv'
vehicle_data_path = '../data/processed/preprocessed_vehicle_data.csv'

# Read the datasets
crashes_df = pd.read_csv(crashes_data_path)
vehicle_df = pd.read_csv(vehicle_data_path)

# Step 1: Merge the datasets on the reference columns ('Name' and 'CrashName')
merged_df = pd.merge(crashes_df, vehicle_df, how='inner', left_on='Name', right_on='CrashName')

# Step 2: Define the severity determination function
def determine_severity(row):
    if row['FatalInjuries'] > 0:
        return 'Fatal'
    elif row['SevereInjuries'] > 0:
        return 'Severe'
    elif row['ModerateInjuries'] > 0:
        return 'Moderate'
    elif row['MinorInjuries'] > 0:
        return 'Minor'
    else:
        return 'NoInjury'

# Step 3: Apply the function to create the 'Severity' column
merged_df['Severity'] = merged_df.apply(determine_severity, axis=1)

# Step 4: Map severity levels to numerical labels
severity_mapping = {'NoInjury': 0, 'Minor': 1, 'Moderate': 2, 'Severe': 3, 'Fatal': 4}
merged_df['Severity_Code'] = merged_df['Severity'].map(severity_mapping)

# Step 5: Prepare features (X) and target variable (y)
# Include the new fields: 'PrimaryCollisionFactor_Code', 'CollisionType_Code', 'HitAndRunFlag'
X = merged_df[['PartyType_Code', 'Sobriety_Code', 'Age', 'SpeedingFlag', 'PrimaryCollisionFactor_Code', 'CollisionType_Code', 'HitAndRunFlag']]
y = merged_df['Severity_Code']

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratify to preserve class distribution
)

# Step 7: Save the splits
# Save X_train and X_test without headers
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)

# Save y_train and y_test with headers
y_train.to_csv('../data/processed/y_train.csv', index=False, header=True)
y_test.to_csv('../data/processed/y_test.csv', index=False, header=True)

# Step 8: Verify alignment of indices
print("Are indices aligned between X_test and y_test?", X_test.index.equals(y_test.index))
print("Data merging and splitting completed successfully!")
