import pandas as pd

# Mappings
sobriety_mapping = {
    "Under Drug Influence": 8,
    "Had Been Drinking - Under Influence": 7,
    "Had Been Drinking - Impairment Unknown": 6,
    "Had Been Drinking - Not Under Influence": 5,
    "Sleepy/Fatigued": 4,
    "Impairment Physical": 3,
    "Impairment Not Known": 2,
    "Had Not Been Drinking": 1,
    "Not Applicable": 0
}
party_type_mapping = {
    "Bicycle": 0,
    "Bus - Other": 1,
    "Bus - School": 2,
    "Car": 3,
    "Car With Trailer": 4,
    "Construction Equipment": 5,
    "Emergency Vehicle": 6,
    "Light Rail Vehicle": 7,
    "Motorcycle/Moped": 8,
    "Panel Truck": 9,
    "Pedestrian": 10,
    "Scooter Motorized": 11,
    "Scooter Non-Motorized": 12,
    "Semi Truck": 13,
    "Skateboard": 14,
    "Train": 15,
    "Wheelchair": 16,
    "Other": 17,
    "Unknown": 18
}
primary_collision_factor_mapping = {
    "Bike At Fault": 0,
    "Other Improper Driving": 1,
    "Other Than Driver": 2,
    "Pedestrian At Fault": 3,
    "Unknown": 4,
    "Violation Driver 1": 5,
    "Violation Driver 2": 6
}
collision_type_mapping = {
    "Broadside": 0,
    "Head On": 1,
    "Hit Object": 2,
    "Other": 3,
    "Overturned": 4,
    "Rear End": 5,
    "Sideswipe": 6,
    "Vehicle/Bike": 7,
    "Vehicle/Pedestrian": 8
}
vehicle_damage_mapping = {
    "None": 0,
    "Minor": 1,
    "Moderate": 2,
    "Major": 3,
    "Totaled": 4,
    "Unknown": 5
}
movement_preceding_collision_mapping = {
    "Proceeding Straight": 0,
    "Making Right Turn": 1,
    "Making Left Turn": 2,
    "Backing": 3,
    "Parking Maneuver": 4,
    "Changing Lanes": 5,
    "Overtaking/Passing": 6,
    "U-Turn": 7,
    "Stopped": 8,
    "Other": 9,
    "Unknown": 10
}

# Preprocessing for crash data
def preprocess_crashes_data(df):
    # Derive 'CrashTime' if 'CrashDateTime' exists
    
    if 'CrashDateTime' in df.columns:
        df['CrashTime'] = pd.to_datetime(df['CrashDateTime'], errors='coerce', infer_datetime_format=True).dt.hour
        # Handle missing or invalid values
        df['CrashTime'] = df['CrashTime'].fillna(0).astype(int)
    else:
        raise KeyError("Neither 'CrashTime' nor 'CrashDateTime' columns are available in the dataset.")
    df['CrashTime'] = df['CrashTime'].fillna(0).astype(int)

    # Map 'PrimaryCollisionFactor' to 'PrimaryCollisionFactor_Code'
    if 'PrimaryCollisionFactor' in df.columns:
        df['PrimaryCollisionFactor_Code'] = df['PrimaryCollisionFactor'].map(primary_collision_factor_mapping).fillna(4).astype(int)
    else:
        raise KeyError("'PrimaryCollisionFactor' column is missing in the dataset.")

    # Map 'CollisionType' to 'CollisionType_Code'
    if 'CollisionType' in df.columns:
        df['CollisionType_Code'] = df['CollisionType'].map(collision_type_mapping).fillna(3).astype(int)
    else:
        raise KeyError("'CollisionType' column is missing in the dataset.")

    # Ensure 'Distance' has no missing values
    df['Distance'] = df['Distance'].fillna(0).astype(float)

    # Select required columns
    key_attributes = [
        'CrashFactId', 'Name', 'MinorInjuries', 'ModerateInjuries', 'SevereInjuries',
        'FatalInjuries', 'PrimaryCollisionFactor_Code', 'CollisionType_Code', 'Distance', 'CrashTime'
    ]
    return df[key_attributes]

# Preprocessing for vehicle crash data
def preprocess_vehicle_crashes_data(df):
    # Map and ensure no blanks
    df['Sobriety_Code'] = df['Sobriety'].map(sobriety_mapping).fillna(0).astype(int)
    df['PartyType_Code'] = df['PartyType'].map(party_type_mapping).fillna(18).astype(int)
    df['VehicleDamage_Code'] = df['VehicleDamage'].map(vehicle_damage_mapping).fillna(5).astype(int)
    df['MovementPrecedingCollision_Code'] = df['MovementPrecedingCollision'].map(movement_preceding_collision_mapping).fillna(10).astype(int)
    # Handle 'ViolationCode': Replace invalid values with 0 and convert to numeric
    df['ViolationCode'] = df['ViolationCode'].replace(['Unknown', 'Not Applicable'], '0').fillna('0').astype(int)
    df['Age'] = df['Age'].fillna(0).astype(int)  # Ensure 'Age' has no blanks
    # Select required columns
    key_attributes = [
        'CrashName', 'PartyType_Code', 'Age', 'Sobriety_Code', 'VehicleDamage_Code',
        'MovementPrecedingCollision_Code', 'ViolationCode'
    ]
    return df[key_attributes]

# Load datasets
crashes_2011_2021 = pd.read_csv('../data/raw/crashdata2011-2021.csv', low_memory=False)
vehicles_2011_2021 = pd.read_csv('../data/raw/vehiclecrashdata2011-2021.csv', low_memory=False)
crashes_2022_present = pd.read_csv('../data/raw/crashdata2022-present.csv', low_memory=False)
vehicles_2022_present = pd.read_csv('../data/raw/vehiclecrashdata2022-present.csv', low_memory=False)

# Apply preprocessing
crashes_combined = pd.concat([
    preprocess_crashes_data(crashes_2011_2021),
    preprocess_crashes_data(crashes_2022_present)
], ignore_index=True)
vehicles_combined = pd.concat([
    preprocess_vehicle_crashes_data(vehicles_2011_2021),
    preprocess_vehicle_crashes_data(vehicles_2022_present)
], ignore_index=True)

# Save processed data
crashes_combined.to_csv('../data/processed/processed_crashes.csv', index=False)
vehicles_combined.to_csv('../data/processed/processed_vehicles.csv', index=False)

print("Preprocessing complete, files saved.")
