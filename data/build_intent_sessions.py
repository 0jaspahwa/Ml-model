# build_intent_sessions.py

import pandas as pd
import numpy as np

# Step 1: Load datasets
activity_df = pd.read_csv("User_Activity_Logs.csv", dtype=str, skipinitialspace=True)
transaction_df = pd.read_csv("Transaction_Dataset.csv")

# Step 2: Normalize column names
activity_df.columns = activity_df.columns.str.strip().str.lower()
transaction_df.columns = transaction_df.columns.str.strip().str.lower()

# ✅ Debug user IDs
print("🧪 Checking user_pseudo_id sample:")
print(activity_df['user_pseudo_id'].dropna().unique()[:5])
print("🧮 Count of valid IDs:", activity_df['user_pseudo_id'].notna().sum())

# Debug: Print column structure and sample rows
print("📊 Activity columns:", activity_df.columns.tolist())
print("📊 Transaction columns:", transaction_df.columns.tolist())
print("🔍 Sample activity rows:")
print(activity_df.head())

# Step 3: Convert timestamps (if needed)
print("🧪 Rows before timestamp filter:", len(activity_df))
activity_df['eventtimestamp'] = pd.to_numeric(activity_df['eventtimestamp'], errors='coerce')
print("🧪 NaNs in eventtimestamp:", activity_df['eventtimestamp'].isna().sum())
activity_df = activity_df.dropna(subset=['eventtimestamp'])
print("✅ Rows after dropping bad timestamps:", len(activity_df))
activity_df['eventtimestamp'] = activity_df['eventtimestamp'].astype(int)

# Step 4: Map events to view/cart/purchase
activity_df['event'] = activity_df['event_name'].map({
    'page_view': 'view',
    'view_item': 'view',
    'product_view': 'view',
    'session_start': 'view',  # Treat session start as a view
    'add_to_cart': 'add_to_cart',
    'begin_checkout': 'add_to_cart',
    'purchase': 'purchase',
    'ecommerce_purchase': 'purchase'
}).fillna('other')

# Step 5: Aggregate session features per user
sessions = []

for user_id, group in activity_df.groupby('user_pseudo_id'):
    group = group.sort_values('eventtimestamp')
    total_views = (group['event'] == 'view').sum()
    total_add_to_cart = (group['event'] == 'add_to_cart').sum()
    total_purchases = (group['event'] == 'purchase').sum()
    last_event_type = group.iloc[-1]['event'] if not group.empty else 'none'

    if len(group) >= 2:
        time_since_last = (group.iloc[-1]['eventtimestamp'] - group.iloc[-2]['eventtimestamp']) / 1000.0
    else:
        time_since_last = np.nan

    # DEBUG: Print user and event counts
    print("👀 User:", user_id)
    print(group[['event_name', 'event']].value_counts())

    # Label intent
    has_purchase = total_purchases > 0 or group['transaction_id'].isin(transaction_df['transaction_id']).any()
    if has_purchase:
        label = "purchase_intent"
    elif total_add_to_cart > 0:
        label = "deal_hunting"
    elif total_views > 2:
        label = "product_comparison"
    else:
        label = "browsing"

    # Metadata
    first_row = group.iloc[0]
    region = first_row.get('region', 'unknown')
    device = first_row.get('category', 'unknown')

    sessions.append({
        "user_id": user_id,
        "device_type": device,
        "region": region,
        "total_views": total_views,
        "total_add_to_cart": total_add_to_cart,
        "total_purchases": total_purchases,
        "last_event_type": last_event_type,
        "time_since_last_event": round(time_since_last, 2) if not pd.isna(time_since_last) else -1,
        "intent_label": label
    })

# Step 6: Save to CSV
session_df = pd.DataFrame(sessions)
print("✅ Generated", len(session_df), "user sessions")
session_df.to_csv("intent_sessions.csv", index=False)
print("✅ Saved intent_sessions.csv with", len(session_df), "rows")
