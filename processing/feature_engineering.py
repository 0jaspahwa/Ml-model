import numpy as np
import pandas as pd
from core.segmentation import assign_behavioral_clusters_fixed

def create_realistic_ga4_data():
    users = []
    for user_id in range(1000):
        age = np.random.choice([22, 28, 35, 42, 55], p=[0.2, 0.25, 0.3, 0.15, 0.1])
        revenue = np.random.uniform(50, 1000)
        events = np.random.randint(10, 100)
        views = int(events * 0.5)
        item_views = int(events * 0.3)
        carts = np.random.randint(0, 10)
        purchases = np.random.randint(0, 5)
        users.append({
            "user_id": f"user_{user_id}",
            "total_events": events,
            "page_views": views,
            "item_views": item_views,
            "cart_additions": carts,
            "purchases": purchases,
            "total_revenue": revenue,
            "engagement_score": (views * 0.1 + item_views * 0.3 + carts * 0.7 + purchases * 1.0) / max(events, 1),
            "conversion_rate": purchases / max(events, 1),
            "purchase_intent_score": carts / max(events, 1),
            "loyalty_score": min(purchases / 3, 1.0),
            "value_score": min(revenue / 500, 1.0),
            "ml_cluster": np.random.randint(0, 5),
            "age_group": str(age),
            "traffic_source": np.random.choice(['Google', 'Email']),
            "device_type": np.random.choice(['mobile', 'desktop'])
        })
    return pd.DataFrame(users)

def create_realistic_transaction_data(users=None):
    if users is None or users.empty:
        return pd.DataFrame()
    transactions = []
    for _, u in users.iterrows():
        for _ in range(u["purchases"]):
            transactions.append({
                "user_id": u["user_id"],
                "Transaction_ID": f"txn_{np.random.randint(1, 100000)}",
                "amount": u["total_revenue"] / max(u["purchases"], 1),
                "Item_revenue": u["total_revenue"] / max(u["purchases"], 1)
            })
    return pd.DataFrame(transactions)

def process_ga4_activity_logs_fixed(activity_df, transaction_df):
    if activity_df.empty:
        return create_realistic_ga4_data()

    df = activity_df.groupby('user_pseudo_id').agg({
        'event_name': 'count',
        'purchase_revenue': 'sum'
    }).reset_index().rename(columns={
        'user_pseudo_id': 'user_id',
        'event_name': 'total_events',
        'purchase_revenue': 'total_revenue'
    })

    df['page_views'] = df['total_events'] * 0.6
    df['item_views'] = df['total_events'] * 0.3
    df['cart_additions'] = df['total_events'] * 0.1
    df['purchases'] = (df['total_revenue'] > 0).astype(int)
    df['engagement_score'] = (
        df['page_views'] * 0.1 +
        df['item_views'] * 0.3 +
        df['cart_additions'] * 0.7 +
        df['purchases'] * 1.0
    ) / df['total_events']
    df['conversion_rate'] = df['purchases'] / df['total_events']
    df['purchase_intent_score'] = df['cart_additions'] / df['total_events']
    df['loyalty_score'] = np.minimum(df['purchases'] / 3, 1.0)
    df['value_score'] = np.minimum(df['total_revenue'] / 500, 1.0)
    df['ml_cluster'] = assign_behavioral_clusters_fixed(df)
    return df
