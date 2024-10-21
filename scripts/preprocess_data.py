import pandas as pd

date_format = '%m/%d/%Y %I:%M:%S %p'
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
# Preprocess function for 'Crashes 2022 - Present'
def preprocess_crashes_data_key_attributes(df):
    key_attributes = ['CrashFactId', 'Name', 'MinorInjuries', 'ModerateInjuries', 'SevereInjuries', 'FatalInjuries', 'CrashDateTime', 'SpeedingFlag']

    df = df.loc[:, key_attributes]
    
    # Crash Time: Extract only the time from CrashDateTime
    df['CrashTime'] = pd.to_datetime(df['CrashDateTime'], format=date_format, errors='coerce').dt.strftime('%H:%M')
    df.drop(columns=['CrashDateTime'], inplace=True)
    
    # Speeding Flag: Ensure binary format
    df['SpeedingFlag'] = df['SpeedingFlag'].apply(lambda x: 1 if x == True else 0)
    
    return df

# Preprocess function for 'Vehicles 2022 - Present'
def preprocess_vehicle_crash_data_key_attributes(df): 
    key_attributes = ['CrashName', 'PartyType', 'Age', 'Sobriety']  # Adjust as per your actual columns
    
    df = df.loc[:, key_attributes]
    
    # Driver Age: Handle missing values, and categorize into groups
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['AgeGroup'] = pd.cut(df['Age'], bins=[-1, 18, 30, 45, 60, 1000], labels=['<18', '18-30', '31-45', '46-60', '60+'])
    

    
    # Apply the mapping to the 'Sobriety' column
    df['Sobriety_Code'] = df['Sobriety'].map(sobriety_mapping)
    
    # Apply the mapping to the 'Party Type' column
    df['PartyType_Code'] = df['PartyType'].map(party_type_mapping)
    
    return df

# Load datasets (adjust file paths as needed)
crashes_df = pd.read_csv('data/raw/crashdata2022-present.csv')
vehicle_crashes_df = pd.read_csv('data/raw/vehiclecrashdata2022-present.csv')

# Applying the updated preprocessing functions
preprocessed_crashes_data_key = preprocess_crashes_data_key_attributes(crashes_df)
preprocessed_vehicle_data_key = preprocess_vehicle_crash_data_key_attributes(vehicle_crashes_df)

# Save preprocessed data
preprocessed_crashes_data_key.to_csv('data/processed/preprocessed_crashes_data.csv', index=False)
preprocessed_vehicle_data_key.to_csv('data/processed/preprocessed_vehicle_data.csv', index=False)

print("Preprocessing complete, files saved in 'data/processed/'")
