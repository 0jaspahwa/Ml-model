import numpy as np

class AdvancedMLSegmentation:
    def __init__(self, user_features_with_ml):
        self.user_features_with_ml = user_features_with_ml or []
        self.segmentation_results = {}
        if not user_features_with_ml.empty:
            self.create_segments()

    def create_segments(self):
        df = self.user_features_with_ml
        try:
            self.segmentation_results = {
                'high_value': df[(df['total_revenue'] > 400) & (df['purchases'] >= 2)],
                'young_explorers': df[df['age_group'].astype(str).str.startswith('2') & (df['item_views'] >= 5)],
                'search_buyers': df[(df['traffic_source'] == 'Google') & (df['purchases'] >= 1)],
                'intent_browsers': df[df['cart_additions'] >= 2],
                'casual_visitors': df[(df['purchases'] == 0) & (df['cart_additions'] == 0)]
            }
        except Exception:
            self.segmentation_results = {}

    def get_user_segment(self, user_id):
        for name, seg in self.segmentation_results.items():
            if user_id in seg['user_id'].values:
                return {'segment': name, 'confidence': 0.8}
        return {'segment': 'general', 'confidence': 0.5}


def assign_behavioral_clusters_fixed(df):
    df = df.copy()
    df['score'] = (
        df['total_revenue'] * 1000 +
        df['total_events'] * 10 +
        df['engagement_score'] * 100 +
        df['purchases'] * 5000 +
        df.index * 0.001
    )
    df_sorted = df.sort_values('score', ascending=False).reset_index(drop=True)
    total = len(df)
    clusters = []
    for i in range(total):
        if i < total * 0.1:
            clusters.append(0)
        elif i < total * 0.25:
            clusters.append(1)
        elif i < total * 0.45:
            clusters.append(2)
        elif i < total * 0.70:
            clusters.append(3)
        else:
            clusters.append(4)
    mapping = dict(zip(df_sorted.index, clusters))
    return [mapping[i] for i in df.index]
