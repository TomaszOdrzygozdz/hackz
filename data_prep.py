import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.api.types import CategoricalDtype

DATA_DIR = './public_data/'
DUMP_DIR = './data/'
TRAIN_FILE = DUMP_DIR + 'train_simple.csv'
TEST_FILE = DUMP_DIR + 'test_simple.csv'
PCA_TRAIN = DUMP_DIR + 'test_pca'
FINAL_OUTPUT = DUMP_DIR + 'final_output.csv'

target = 'damage_grade'
ids_columns = ['building_id', 'district_id', 'vdcmun_id', 'ward_id']
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
statistics_for = categorical_columns

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
    scaler = StandardScaler()
    train_df_scaled = scaler.fit_transform(
        train_df[[column_name]].values.reshape(-1, 1))
    test_df_scaled = scaler.transform(
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

    for feature_name in ids_columns:
        train_df, test_df = find_statistics(feature_name, train_df, test_df, True)
    for feature_name in statistics_for:
        train_df, test_df = find_statistics(feature_name, train_df, test_df, False)

    uuu = train_df.columns
    #FEATURES engineering
    train_df['neighbours'] = 'Yes'
    train_df.loc[train_df['position'] == 'Not attached', 'neighbours'] = 'No'
    test_df['neighbours'] = 'Yes'
    test_df.loc[train_df['position'] == 'Not attached', 'neighbours'] = 'No'
    # new_categorical_cols = ['neighbours']
    categorical_columns.append('neighbours')

    train_df['simple_plan_configuration'] = 'No'
    train_df.loc[train_df['plan_configuration'] == 'Rectangular', 'simple_plan_configuration'] = 'Yes'
    train_df.loc[train_df['plan_configuration'] == 'Square', 'simple_plan_configuration'] = 'Yes'
    test_df['simple_plan_configuration'] = 'No'
    test_df.loc[test_df['plan_configuration'] == 'Rectangular', 'simple_plan_configuration'] = 'Yes'
    test_df.loc[test_df['plan_configuration'] == 'Square', 'simple_plan_configuration'] = 'Yes'
    categorical_columns.append('simple_plan_configuration')
    # new_categorical_cols.append('simple_plan_configuration')

    train_df['more_than_two_floors'] = 'No'
    test_df['more_than_two_floors'] = 'No'
    train_df.loc[train_df['count_floors_pre_eq'] > 2, 'more_than_two_floors'] = 'Yes'
    test_df.loc[test_df['count_floors_pre_eq'] > 2, 'more_than_two_floors'] = 'Yes'
    categorical_columns.append('more_than_two_floors')

    train_df['number_of_geotechnical_risks'] = train_df[
        ['has_geotechnical_risk_fault_crack', 'has_geotechnical_risk_flood',
         'has_geotechnical_risk_land_settlement', 'has_geotechnical_risk_landslide',
         'has_geotechnical_risk_liquefaction', 'has_geotechnical_risk_other',
         'has_geotechnical_risk_rock_fall']].sum(axis=1)
    train_df['number_of_geotechnical_risks_higher_than_0'] = 0
    train_df.loc[train_df['number_of_geotechnical_risks'] > 0, 'number_of_geotechnical_risks_higher_than_0'] = 1
    train_df['number_of_geotechnical_risks_higher_than_1'] = 0
    train_df.loc[train_df['number_of_geotechnical_risks'] > 1, 'number_of_geotechnical_risks_higher_than_1'] = 1
    train_df['number_of_geotechnical_risks_higher_than_2'] = 0
    train_df.loc[train_df['number_of_geotechnical_risks'] > 2, 'number_of_geotechnical_risks_higher_than_2'] = 1
    train_df['number_of_geotechnical_risks_higher_than_3'] = 0
    train_df.loc[train_df['number_of_geotechnical_risks'] > 3, 'number_of_geotechnical_risks_higher_than_3'] = 1
    train_df['number_of_geotechnical_risks_higher_than_4'] = 0
    train_df.loc[train_df['number_of_geotechnical_risks'] > 4, 'number_of_geotechnical_risks_higher_than_4'] = 1
    train_df['number_of_geotechnical_risks_higher_than_5'] = 0
    train_df.loc[train_df['number_of_geotechnical_risks'] > 5, 'number_of_geotechnical_risks_higher_than_5'] = 1
    test_df['number_of_geotechnical_risks'] = test_df[
        ['has_geotechnical_risk_fault_crack', 'has_geotechnical_risk_flood',
         'has_geotechnical_risk_land_settlement', 'has_geotechnical_risk_landslide',
         'has_geotechnical_risk_liquefaction', 'has_geotechnical_risk_other',
         'has_geotechnical_risk_rock_fall']].sum(axis=1)
    test_df['number_of_geotechnical_risks_higher_than_0'] = 0
    test_df.loc[test_df['number_of_geotechnical_risks'] > 0, 'number_of_geotechnical_risks_higher_than_0'] = 1
    test_df['number_of_geotechnical_risks_higher_than_1'] = 0
    test_df.loc[test_df['number_of_geotechnical_risks'] > 1, 'number_of_geotechnical_risks_higher_than_1'] = 1
    test_df['number_of_geotechnical_risks_higher_than_2'] = 0
    test_df.loc[test_df['number_of_geotechnical_risks'] > 2, 'number_of_geotechnical_risks_higher_than_2'] = 1
    test_df['number_of_geotechnical_risks_higher_than_3'] = 0
    test_df.loc[test_df['number_of_geotechnical_risks'] > 3, 'number_of_geotechnical_risks_higher_than_3'] = 1
    test_df['number_of_geotechnical_risks_higher_than_4'] = 0
    test_df.loc[test_df['number_of_geotechnical_risks'] > 4, 'number_of_geotechnical_risks_higher_than_4'] = 1
    test_df['number_of_geotechnical_risks_higher_than_5'] = 0
    test_df.loc[test_df['number_of_geotechnical_risks'] > 5, 'number_of_geotechnical_risks_higher_than_5'] = 1
    categorical_columns.append('number_of_geotechnical_risks')


    # df = pd.DataFrame()
    # for i in range(1, 6):
    #     df['no_of_{}_in_district'.format(i)] = train_df.groupby('district_id')[target].value_counts().unstack()[i]

    # ONE HOT ENCODING AND SCALING
    for n in numerical_columns:
        train_df, test_df = get_scaled_column(n, train_df, test_df)

    for c in categorical_columns:
        train_df, test_df = get_dummies_from_value_in_column(c, train_df,
                                                             test_df)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    maxes = train_df.max()

def balance_dataset(train_df):
    balance_length = train_df.groupby(target).count().min()[0]
    # balance_length
    # grade_1
    b1 = train_df[train_df[target] == 1].sample(n=balance_length)
    b2 = train_df[train_df[target] == 2].sample(n=balance_length)
    b3 = train_df[train_df[target] == 3].sample(n=balance_length)
    b4 = train_df[train_df[target] == 4].sample(n=balance_length)
    b5 = train_df[train_df[target] == 5].sample(n=balance_length)
    balanced_train_df = pd.concat([b1, b2, b3, b4, b5]).copy()
    return balanced_train_df

def load_train():
    return pd.read_csv(TRAIN_FILE)

def load_test():
    return pd.read_csv(TEST_FILE)

def save_final_output(df):
    df.to_csv(FINAL_OUTPUT, index=False)

def load_X_Y(df):
    X, Y = df.loc[:, df.columns != target], df[[target]]
    return X, Y

def load_X_Y_file(file_name):
    df = pd.read_csv(DUMP_DIR + file_name + '.csv')
    return load_X_Y(df)

def remove_cols(col_list, file_name):
    df = load_train()
    df = df.drop(columns=col_list)
    df.to_csv(DUMP_DIR + file_name, index=False)

def check_cols():
    print(len([target] + ids_columns + categorical_columns + onehot_columns + numerical_columns))

def dump_predictions(X_test_id, output_):
    df_to_save = pd.DataFrame()
    df_to_save['building_id'] = X_test_id
    df_to_save[target] = output_
    save_final_output(df_to_save)

def find_statistics(feature_name, train_df, test_df, drop=False):
    numerical_columns.append(f'mean_damage_grade_for_{feature_name}')
    mean_damage_grade_for_district_id = train_df.groupby(feature_name)[target].mean()
    df = pd.DataFrame()
    df[f'mean_damage_grade_for_{feature_name}'] = mean_damage_grade_for_district_id
    for i in range(1, 6):
        df[f'dmg_lvl_{i}_in_{feature_name}'] = train_df.groupby(feature_name)[target].value_counts().unstack()[i]
    df = df.div(df.sum(axis=1), axis=0)
    train_df = pd.merge(train_df, df, left_on=feature_name, right_index=True, how='left')
    test_df = pd.merge(test_df, df, left_on=feature_name, right_index=True, how='left')
    if drop:
        train_df.drop(columns=[feature_name], inplace=True)
        test_df.drop(columns=[feature_name], inplace=True)
    return train_df, test_df