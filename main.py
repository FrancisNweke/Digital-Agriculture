import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils.np_utils import to_categorical
from custom_models.net_model import agro_lgbm, agro_mlp


# Load dataset
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data['Train_Flag'] = 1
test_data['Train_Flag'] = 0
test_data['Crop_Damage'] = 0

merged_dataset = pd.concat((train_data, test_data))

merged_dataset['ID_Value'] = merged_dataset['ID'].apply(lambda x: x.strip('F')).astype('int')
merged_dataset = merged_dataset.sort_values(['ID_Value'])
merged_dataset = merged_dataset.reset_index(drop=True)

# Add more columns
merged_dataset['Soil_Type_Damage'] = merged_dataset.sort_values(['ID_Value']).groupby(['Soil_Type'])[
    'Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values
merged_dataset['Estimated_Insects_Count_Damage'] = \
    merged_dataset.sort_values(['ID_Value']).groupby(['Estimated_Insects_Count'])['Crop_Damage'].apply(
        lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values
merged_dataset['Crop_Type_Damage'] = merged_dataset.sort_values(['ID_Value']).groupby(['Crop_Type'])[
    'Crop_Damage'].apply(lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values
merged_dataset['Pesticide_Use_Category_Damage'] = \
    merged_dataset.sort_values(['ID_Value']).groupby(['Pesticide_Use_Category'])['Crop_Damage'].apply(
        lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values
merged_dataset['Season_Damage'] = merged_dataset.sort_values(['ID_Value']).groupby(['Season'])['Crop_Damage'].apply(
    lambda x: x.shift().rolling(5, min_periods=1).mean()).fillna(-999).values

merged_dataset['Soil_Type_Damage_c2'] = merged_dataset.sort_values(['ID_Value']).groupby(['Soil_Type'])[
    'Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values
merged_dataset['Estimated_Insects_Count_Damage_c2'] = \
    merged_dataset.sort_values(['ID_Value']).groupby(['Estimated_Insects_Count'])['Crop_Damage'].apply(
        lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values
merged_dataset['Crop_Type_Damage_c2'] = merged_dataset.sort_values(['ID_Value']).groupby(['Crop_Type'])[
    'Crop_Damage'].apply(lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values
merged_dataset['Pesticide_Use_Category_Damage_c2'] = \
    merged_dataset.sort_values(['ID_Value']).groupby(['Pesticide_Use_Category'])['Crop_Damage'].apply(
        lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values
merged_dataset['Season_Damage_c2'] = merged_dataset.sort_values(['ID_Value']).groupby(['Season'])['Crop_Damage'].apply(
    lambda x: x.shift(periods=2).rolling(5, min_periods=1).mean()).fillna(-999).values

merged_dataset.loc[merged_dataset['Train_Flag'] == 0, 'Crop_Damage'] = -999

# Add more feature columns
merged_dataset['Crop_Damage_c1'] = merged_dataset['Crop_Damage'].shift(fill_value=-999)
merged_dataset['Estimated_Insects_Count_c1'] = merged_dataset['Estimated_Insects_Count'].shift(fill_value=-999)
merged_dataset['Crop_Type_c1'] = merged_dataset['Crop_Type'].shift(fill_value=-999)
merged_dataset['Soil_Type_c1'] = merged_dataset['Soil_Type'].shift(fill_value=-999)
merged_dataset['Pesticide_Use_Category_c1'] = merged_dataset['Pesticide_Use_Category'].shift(fill_value=-999)
merged_dataset['Number_Doses_Week_c1'] = merged_dataset['Number_Doses_Week'].shift(fill_value=-999)
merged_dataset['Number_Weeks_Used_c1'] = merged_dataset['Number_Weeks_Used'].shift(fill_value=-999)
merged_dataset['Number_Weeks_Quit_c1'] = merged_dataset['Number_Weeks_Quit'].shift(fill_value=-999)
merged_dataset['Season_c1'] = merged_dataset['Season'].shift(fill_value=-999)

merged_dataset['Crop_Damage_c2'] = merged_dataset['Crop_Damage'].shift(periods=2, fill_value=-999)
merged_dataset['Estimated_Insects_Count_c2'] = merged_dataset['Estimated_Insects_Count'].shift(periods=2,
                                                                                               fill_value=-999)
merged_dataset['Crop_Type_c2'] = merged_dataset['Crop_Type'].shift(periods=2, fill_value=-999)
merged_dataset['Soil_Type_c2'] = merged_dataset['Soil_Type'].shift(periods=2, fill_value=-999)
merged_dataset['Pesticide_Use_Category_c2'] = merged_dataset['Pesticide_Use_Category'].shift(periods=2, fill_value=-999)
merged_dataset['Number_Doses_Week_c2'] = merged_dataset['Number_Doses_Week'].shift(periods=2, fill_value=-999)
merged_dataset['Number_Weeks_Used_c2'] = merged_dataset['Number_Weeks_Used'].shift(periods=2, fill_value=-999)
merged_dataset['Number_Weeks_Quit_c2'] = merged_dataset['Number_Weeks_Quit'].shift(periods=2, fill_value=-999)
merged_dataset['Season_c2'] = merged_dataset['Season'].shift(periods=2, fill_value=-999)

# Split data into train and test datasets
train_dataset, test_dataset = merged_dataset[merged_dataset.Train_Flag == 1], merged_dataset[
    merged_dataset.Train_Flag == 0]

train_dt = train_dataset.drop(columns=['Train_Flag'], axis=11)
test_dt_1 = test_dataset.drop(columns=['Train_Flag'], axis=11)
test_dt = test_dt_1.drop(columns=['Crop_Damage'], axis=10)

# Delete the merged dataset to save memory
del merged_dataset

missing_value = -999

# Assign values to empty cells
train_dt['Number_Weeks_Used'] = train_dt['Number_Weeks_Used'].apply(lambda x: missing_value if pd.isna(x) else x)
test_dt['Number_Weeks_Used'] = test_dt['Number_Weeks_Used'].apply(lambda x: missing_value if pd.isna(x) else x)

train_dt['Number_Weeks_Used_c1'] = train_dt['Number_Weeks_Used_c1'].apply(lambda x: missing_value if pd.isna(x) else x)
test_dt['Number_Weeks_Used_c1'] = test_dt['Number_Weeks_Used_c1'].apply(lambda x: missing_value if pd.isna(x) else x)

train_dt['Number_Weeks_Used_c2'] = train_dt['Number_Weeks_Used_c2'].apply(lambda x: missing_value if pd.isna(x) else x)
test_dt['Number_Weeks_Used_c2'] = test_dt['Number_Weeks_Used_c2'].apply(lambda x: missing_value if pd.isna(x) else x)

dataset_X = train_dt.drop(columns=['Crop_Damage', 'ID', 'ID_Value'])
dataset_y = train_dt['Crop_Damage'].values

# Remove the ID and ID_Value columns from test dataset
test_dt.drop(columns=['ID', 'ID_Value'], inplace=True)

# dataset_X = np.array(dataset_X)
# dataset_y = to_categorical(dataset_y, num_classes=3)

# Split data into train and test
train_X, test_X, train_y, test_y = train_test_split(dataset_X, dataset_y, random_state=42, test_size=0.3, shuffle=True)

# Split data into train and validation
train_X1, eval_X, train_y1, eval_y = train_test_split(train_X, train_y, random_state=42, test_size=0.25, shuffle=True)

# Number of output classes: (0=alive, 1=Damage due to other causes, 2=Damage due to Pesticides)
num_classes = 3

# Using LightGBM
# agro_lgbm(train_X1, train_y1, eval_X, eval_y, test_X, test_y, test_dt)

# Using MLP
# Note: to use MLP, uncomment the code for one-hot encoding. See line 103: to_categorical()
agro_mlp(train_X1, train_y1, eval_X, eval_y, test_X, test_y, test_dt, num_classes)




