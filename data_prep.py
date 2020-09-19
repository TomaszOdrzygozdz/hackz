import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.api.types import CategoricalDtype

DATA_DIR = './public_data/'
DUMP_DIR = 'data/'
TRAIN_FILE = DUMP_DIR + 'train_simple.csv'
TEST_FILE = DUMP_DIR + 'test_simple.csv'

target = 'damage_grade'
ids_columns = ['building_id', 'istrict_id', 'vdcmun_id', 'ward_id']
categorical_columns = ['legal_ownership_status', 'land_surface_condition', 'foundation_type','roof_type',
                       'ground_floor_type', 'other_floor_type', 'position','plan_configuration', 'income_range_in_thousands' ]
onehot_columns = ['has_secondary_use', 'has_secondary_use_agriculture','has_secondary_use_hotel',
                  'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school',
                  'has_secondary_use_industry', 'has_secondary_use_health_post','has_secondary_use_gov_office',
                  'has_secondary_use_use_police','has_secondary_use_other', 'has_superstructure_adobe_mud',
                  'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
                  'has_superstructure_cement_mortar_stone','has_superstructure_mud_mortar_brick',
                  'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
                  'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
                  'has_superstructure_rc_engineered', 'has_superstructure_other','has_geotechnical_risk',
                  'has_geotechnical_risk_fault_crack','has_geotechnical_risk_flood',
                  'has_geotechnical_risk_land_settlement','has_geotechnical_risk_landslide',
                  'has_geotechnical_risk_liquefaction','has_geotechnical_risk_other',
                  'has_geotechnical_risk_rock_fall']
numerical_columns = ['count_families', 'count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft',
                     'height_ft_pre_eq', 'household_count', 'avg_hh_size']


def get_dummies_from_value_in_column(column_name, train_df, test_df):
    train_df[column_name].fillna(value='no_info', inplace=True)
    test_df[column_name].fillna(value='no_info', inplace=True)

    train_column = train_df[[column_name]]
    test_column = test_df[[column_name]]
    all_data = pd.concat([train_column, test_column])

    for column in all_data.select_dtypes(include=[np.object]).columns:
        cat_type = CategoricalDtype(categories=all_data[column].unique(),
                                    ordered=True)
        train_column[column].astype(cat_type)
        test_column[column].astype(cat_type)

    onehot_train = pd.get_dummies(train_column, prefix=column_name)
    onehot_test = pd.get_dummies(test_column, prefix=column_name)

    train_df.drop(columns=[column_name], inplace=True)
    test_df.drop(columns=[column_name], inplace=True)

    return pd.merge(train_df, onehot_train, left_index=True,
                    right_index=True), pd.merge(test_df, onehot_test,
                                                left_index=True,
                                                right_index=True)


def get_scaled_column(column_name, train_df, test_df):
    scaler = MinMaxScaler()
    train_df_scaled = scaler.fit_transform(
        train_df[[column_name]].values.reshape(-1, 1))
    test_df_scaled = scaler.fit_transform(
        test_df[column_name].values.reshape(-1, 1))
    train_df[column_name] = train_df_scaled
    test_df[column_name] = test_df_scaled
    return train_df, test_df

def prep_data():
    building_ownership = pd.read_csv(DATA_DIR + 'building_ownership.csv')
    building_structure = pd.read_csv(DATA_DIR + 'building_structure.csv')
    train = pd.read_csv(DATA_DIR + 'train.csv')
    test = pd.read_csv(DATA_DIR + 'test.csv')
    ward_demographic_data = pd.read_csv(DATA_DIR + 'ward_demographic_data.csv')
    merged_df = pd.merge(building_ownership, building_structure, on='building_id', how='left')
    merged_df.drop(columns=['district_id_x', 'vdcmun_id_x', 'ward_id_x'],inplace=True)
    merged_df.rename(columns={'district_id_y':'district_id', 'vdcmun_id_y':'vdcmun_id', 'ward_id_y':'ward_id'}, inplace=True)
    merged_df = pd.merge(merged_df, ward_demographic_data, on='ward_id', how='left')
    train_df = pd.merge(merged_df, train, on='building_id', how='right')
    test_df = pd.merge(merged_df, test, on='building_id', how='right')
    # FILL IN MISSING DATA
    train_df = train_df.fillna(train_df.median())
    test_df = test_df.fillna(train_df.median())
    # ONE HOT ENCODING AND SCALING
    for n in numerical_columns:
        train_df, test_df = get_scaled_column(n, train_df, test_df)

    for c in categorical_columns:
        train_df, test_df = get_dummies_from_value_in_column(c, train_df,
                                                             test_df)
    train_df.to_csv(TRAIN_FILE)
    test_df.to_csv(TEST_FILE)

def load_data():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
