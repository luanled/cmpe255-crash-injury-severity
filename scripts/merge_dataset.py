import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed datasets
crashes_df = pd.read_csv('../data/processed/processed_crashes.csv')
vehicles_df = pd.read_csv('../data/processed/processed_vehicles.csv')

# Merge datasets
merged_df = pd.merge(crashes_df, vehicles_df, how='inner', left_on='Name', right_on='CrashName')

# Create severity mapping
severity_mapping = {
    'Fatal': 4, 
    'Severe': 3, 
    'Moderate': 2, 
    'Minor': 1, 
    'NoInjury': 0
}

# Derive Severity
merged_df['Severity'] = merged_df.apply(
    lambda row: 'Fatal' if row['FatalInjuries'] > 0 else 
                'Severe' if row['SevereInjuries'] > 0 else
                'Moderate' if row['ModerateInjuries'] > 0 else
                'Minor' if row['MinorInjuries'] > 0 else
                'NoInjury', axis=1
)
merged_df['Severity_Code'] = merged_df['Severity'].map(severity_mapping)

# Select features and target
features = [
    'PartyType_Code', 'Sobriety_Code', 'Age', 'PrimaryCollisionFactor_Code', 
    'CollisionType_Code', 'VehicleDamage_Code', 'MovementPrecedingCollision_Code', 
    'ViolationCode', 'CrashTime', 'Distance'
]
X = merged_df[features]
y = merged_df['Severity_Code']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False, header=True)
y_test.to_csv('../data/processed/y_test.csv', index=False, header=True)

print("Data merging and splitting completed successfully.")
