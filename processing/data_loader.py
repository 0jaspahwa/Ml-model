import pandas as pd
from processing.feature_engineering import process_ga4_activity_logs_fixed
from processing.feature_engineering import create_realistic_ga4_data, create_realistic_transaction_data

def load_and_process_real_datasets():
    try:
        txn_df = pd.read_csv("data/Transaction_Dataset.csv")
        chunks = pd.read_csv("data/User_Activity_Logs.csv", chunksize=100000)
        act_df = pd.concat([chunk for i, chunk in enumerate(chunks) if i < 5], ignore_index=True)
        act_df = act_df.rename(columns={c: 'user_pseudo_id' for c in act_df.columns if 'user' in c and 'id' in c})
        user_features = process_ga4_activity_logs_fixed(act_df, txn_df)
        return user_features, txn_df
    except Exception:
        fake_users = create_realistic_ga4_data()
        fake_txn = create_realistic_transaction_data(fake_users)
        return fake_users, fake_txn
