from new_data_prep import remove_cols, ids_columns, prep_data, load_train, load_test, find_statistics

prep_data()
X = load_train()
print(len(X.columns))
#remove_cols(ids_columns, 'no_ids.csv'

# train_df = load_train()
# test_df = load_test()
# tr, test = find_statistics('district_id', train_df, test_df)


for nn in X.columns:
    z = X[nn]
    print(f'{nn}: max = {z.max()} min = {z.min()} uniq = {len(z.unique())}')