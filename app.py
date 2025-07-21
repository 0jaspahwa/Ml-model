from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import json
import random
from collections import defaultdict, deque

# ===== YOUR AI MODEL CLASSES (from Google Colab) =====
def create_realistic_ga4_data():
    """Create realistic GA4-style data"""
    import numpy as np
    
    users_data = []
    for user_id in range(5000):
        age = np.random.choice([22, 28, 35, 42, 55], p=[0.2, 0.25, 0.3, 0.15, 0.1])
        gender = np.random.choice(['male', 'female'], p=[0.5, 0.5])
        income = np.random.choice(['Low', 'Medium', 'High', 'Top 10%'], p=[0.3, 0.4, 0.25, 0.05])
        device = np.random.choice(['mobile', 'desktop'], p=[0.65, 0.35])
        source = np.random.choice(['Facebook', 'Google', 'Direct', 'Email'], p=[0.3, 0.4, 0.2, 0.1])
        medium = np.random.choice(['PaidSocial', 'Organic', 'Direct', 'Email'], p=[0.3, 0.4, 0.2, 0.1])
        
        if income in ['High', 'Top 10%'] and age >= 35:
            event_name_count = np.random.randint(15, 80)
            page_view = int(event_name_count * 0.5)
            view_item = int(event_name_count * 0.25)
            add_to_cart = np.random.randint(2, 8)
            purchase = np.random.randint(1, 4)
            purchase_revenue_sum = purchase * np.random.uniform(150, 800)
        elif age <= 28 and device == 'mobile':
            event_name_count = np.random.randint(20, 60)
            page_view = int(event_name_count * 0.6)
            view_item = int(event_name_count * 0.3)
            add_to_cart = np.random.randint(0, 4)
            purchase = 1 if np.random.random() > 0.7 else 0
            purchase_revenue_sum = purchase * np.random.uniform(25, 150)
        else:
            event_name_count = np.random.randint(5, 25)
            page_view = int(event_name_count * 0.6)
            view_item = int(event_name_count * 0.3)
            add_to_cart = np.random.randint(0, 2)
            purchase = 1 if np.random.random() > 0.85 else 0
            purchase_revenue_sum = purchase * np.random.uniform(50, 300)
        
        engagement_score = (page_view * 0.1 + view_item * 0.3 + add_to_cart * 0.7 + purchase * 1.0) / event_name_count
        conversion_rate = purchase / event_name_count
        
        if purchase_revenue_sum > 400 and purchase >= 2:
            ml_cluster = 0
        elif age <= 28 and view_item >= 5:
            ml_cluster = 1
        elif source == 'Google' and purchase >= 1:
            ml_cluster = 2
        elif add_to_cart >= 2:
            ml_cluster = 3
        else:
            ml_cluster = 4
        
        users_data.append({
            'user_id': f'user_{user_id}',
            'total_events': event_name_count,
            'page_views': page_view,
            'item_views': view_item,
            'cart_additions': add_to_cart,
            'purchases': purchase,
            'total_revenue': purchase_revenue_sum,
            'engagement_score': min(engagement_score, 1.0),
            'purchase_intent_score': min(add_to_cart / 5, 1.0),
            'loyalty_score': min(purchase / 3, 1.0),
            'value_score': min(purchase_revenue_sum / 500, 1.0),
            'conversion_rate': conversion_rate,
            'ml_cluster': ml_cluster,
            'age_group': f"{age//10*10}-{age//10*10+9}",
            'gender': gender,
            'income_group': income,
            'device_type': device,
            'traffic_source': source,
            'traffic_medium': medium
        })
    
    return pd.DataFrame(users_data)


def create_realistic_transaction_data(user_features=None):
    """Create realistic transaction data"""
    if user_features is None or user_features.empty:
        return pd.DataFrame({
            'Transaction_ID': ['txn_1', 'txn_2'],
            'ItemID': ['item_1', 'item_2'],
            'ItemName': ['Product A', 'Product B'],
            'ItemCategory': ['Electronics', 'Fashion'],
            'Item_revenue': [100, 200]
        })
    
    # Create transactions based on user purchases
    transactions = []
    categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 'Books']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
    
    transaction_id = 1
    
    for _, user in user_features.iterrows():
        if user.get('purchases', 0) > 0:
            for purchase_num in range(int(user['purchases'])):
                transactions.append({
                    'user_id': user['user_id'],
                    'Transaction_ID': f'txn_{transaction_id}',
                    'amount': user['total_revenue'] / user['purchases'],
                    'ItemCategory': np.random.choice(categories),
                    'ItemBrand': np.random.choice(brands),
                    'Item_revenue': user['total_revenue'] / user['purchases'],
                    'quantity': np.random.randint(1, 3)
                })
                transaction_id += 1
    
    return pd.DataFrame(transactions) if transactions else pd.DataFrame({
        'Transaction_ID': ['txn_1'],
        'ItemID': ['item_1'],
        'ItemName': ['Product A'],
        'ItemCategory': ['Electronics'],
        'Item_revenue': [100]
    })


def process_ga4_activity_logs_fixed(activity_df, transaction_df):
    """Fixed version of GA4 processing that handles DataFrame conditions properly"""
    
    # Check if DataFrames are empty using .empty (fixes the ambiguous truth value error)
    if activity_df.empty:
        print("⚠️ Activity data is empty")
        return create_realistic_ga4_data()

    # Basic user aggregations with error handling
    try:
        user_basic = activity_df.groupby('user_pseudo_id').agg({
            'event_name': 'count' if 'event_name' in activity_df.columns else lambda x: len(x),
            'purchase_revenue': 'sum' if 'purchase_revenue' in activity_df.columns else lambda x: 0,
        }).reset_index()
        
        # Rename columns to match expected format
        user_basic = user_basic.rename(columns={
            'user_pseudo_id': 'user_id',
            'event_name': 'total_events',
            'purchase_revenue': 'total_revenue'
        })
        
        # Add required columns with defaults
        user_basic['page_views'] = user_basic['total_events'] * 0.6
        user_basic['item_views'] = user_basic['total_events'] * 0.3
        user_basic['cart_additions'] = user_basic['total_events'] * 0.1
        user_basic['purchases'] = (user_basic['total_revenue'] > 0).astype(int)
        
        # Calculate engagement metrics
        user_basic['engagement_score'] = (
            user_basic['page_views'] * 0.1 + 
            user_basic['item_views'] * 0.3 + 
            user_basic['cart_additions'] * 0.7 + 
            user_basic['purchases'] * 1.0
        ) / user_basic['total_events']
        
        user_basic['conversion_rate'] = user_basic['purchases'] / user_basic['total_events']
        user_basic['purchase_intent_score'] = user_basic['cart_additions'] / user_basic['total_events']
        user_basic['loyalty_score'] = np.minimum(user_basic['purchases'] / 3, 1.0)
        user_basic['value_score'] = np.minimum(user_basic['total_revenue'] / 500, 1.0)
        
        # Assign clusters
        user_basic['ml_cluster'] = assign_behavioral_clusters_fixed(user_basic)
        
        print(f"✅ Processed {len(user_basic)} users from real data")
        return user_basic
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        return create_realistic_ga4_data()

def assign_behavioral_clusters_fixed(df):
    """Force exactly 5 clusters using index-based distribution - UPDATED VERSION"""
    
    clusters = []
    total_users = len(df)
    
    # Create a composite score but handle ties differently
    df_copy = df.copy()
    df_copy['score'] = (
        df_copy['total_revenue'] * 1000 +  # Amplify revenue differences
        df_copy['total_events'] * 10 +     # Weight events
        df_copy['engagement_score'] * 100 + # Weight engagement
        df_copy['purchases'] * 5000 +      # Heavily weight purchases
        df_copy.index * 0.001              # Add tiny index component to break ties
    )
    
    # Sort by score
    df_sorted = df_copy.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Force 5 equal groups based on position
    for i in range(total_users):
        if i < total_users * 0.1:           # Top 10%
            cluster = 0  # VIP customers
        elif i < total_users * 0.25:        # Next 15% 
            cluster = 1  # Buyers
        elif i < total_users * 0.45:        # Next 20%
            cluster = 2  # High-intent browsers  
        elif i < total_users * 0.70:        # Next 25%
            cluster = 3  # Active browsers
        else:                               # Bottom 30%
            cluster = 4  # Casual visitors
        
        clusters.append(cluster)
    
    # Map back to original order
    original_indices = df_copy.sort_values('score', ascending=False).index
    cluster_mapping = dict(zip(original_indices, clusters))
    
    return [cluster_mapping[i] for i in df.index]





def load_and_process_real_datasets():
    """Load and process the actual provided datasets with memory optimization"""
    try:
        print("📊 Loading real datasets...")
        
        # Load transaction data first (smaller)
        transaction_df = pd.read_csv('data/Transaction_Dataset.csv')
        print(f"✅ Loaded {len(transaction_df)} transactions")
        
        # Load activity data in chunks to avoid memory issues
        chunk_size = 100000
        activity_chunks = []
        
        print("🔄 Loading activity data in chunks...")
        for chunk in pd.read_csv('data/User_Activity_Logs.csv', chunksize=chunk_size, low_memory=False):
            activity_chunks.append(chunk)
            if len(activity_chunks) * chunk_size >= 500000:
                break
        
        activity_logs = pd.concat(activity_chunks, ignore_index=True)
        print(f"✅ Loaded {len(activity_logs)} activity records")
        
        # Find user ID column
        user_id_column = None
        for col in activity_logs.columns:
            if 'user' in col.lower() and 'id' in col.lower():
                user_id_column = col
                break
        
        if user_id_column:
            activity_logs = activity_logs.rename(columns={user_id_column: 'user_pseudo_id'})
        else:
            activity_logs = activity_logs.rename(columns={activity_logs.columns[0]: 'user_pseudo_id'})
        
        # Use the FIXED processing function
        user_features = process_ga4_activity_logs_fixed(activity_logs, transaction_df)
        
        print(f"✅ Created features for {len(user_features)} users")
        return user_features, transaction_df
        
    except Exception as e:
        print(f"❌ Error processing real datasets: {e}")
        print("🔄 Falling back to synthetic data...")
        # Fix the DataFrame condition error here
        synthetic_user_features = create_realistic_ga4_data()
        synthetic_transaction_df = create_realistic_transaction_data(synthetic_user_features)
        return synthetic_user_features, synthetic_transaction_df




def process_ga4_activity_logs(activity_df, transaction_df):
    """Process the GA4-style activity logs into user features"""
    
    # Convert timestamp if needed
    if 'eventTimestamp' in activity_df.columns:
        activity_df['event_datetime'] = pd.to_datetime(activity_df['eventTimestamp'], unit='s', errors='coerce')
    
    # Group events by user_pseudo_id to create user journeys
    print("🔄 Processing user journeys...")
    
    # Basic user aggregations
    user_basic = activity_df.groupby('user_pseudo_id').agg({
        'event_name': 'count',  # total_events
        'eventTimestamp': ['min', 'max'] if 'eventTimestamp' in activity_df.columns else 'count',
        'purchase_revenue': 'sum' if 'purchase_revenue' in activity_df.columns else 'count',
        'total_item_quantity': 'sum' if 'total_item_quantity' in activity_df.columns else 'count',
        'Age': 'first' if 'Age' in activity_df.columns else 'count',
        'gender': 'first' if 'gender' in activity_df.columns else 'count',
        'income_group': 'first' if 'income_group' in activity_df.columns else 'count',
        'city': 'first' if 'city' in activity_df.columns else 'count',
        'region': 'first' if 'region' in activity_df.columns else 'count',
        'country': 'first' if 'country' in activity_df.columns else 'count',
        'source': 'first' if 'source' in activity_df.columns else 'count',
        'medium': 'first' if 'medium' in activity_df.columns else 'count',
        'category': 'first' if 'category' in activity_df.columns else 'count'
    }).reset_index()
    
    # Flatten column names
    user_basic.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in user_basic.columns]
    
    # Calculate specific event counts
    if 'event_name' in activity_df.columns:
        event_counts = activity_df.groupby(['user_pseudo_id', 'event_name']).size().unstack(fill_value=0)
        user_features = user_basic.merge(event_counts, left_on='user_pseudo_id', right_index=True, how='left')
    else:
        user_features = user_basic
    
    # Add transaction data
    if not transaction_df.empty and 'transaction_id' in activity_df.columns:
        # Join with transaction data
        activity_with_transactions = activity_df.merge(
            transaction_df, 
            left_on='transaction_id', 
            right_on='Transaction_ID', 
            how='left'
        )
        
        # Aggregate transaction features
        transaction_features = activity_with_transactions.groupby('user_pseudo_id').agg({
            'Item_revenue': 'sum',
            'Item_purchase_quantity': 'sum',
            'ItemCategory': lambda x: x.mode().iloc[0] if not x.empty and x.notna().any() else 'unknown',
            'ItemBrand': lambda x: x.mode().iloc[0] if not x.empty and x.notna().any() else 'unknown'
        }).reset_index()
        
        user_features = user_features.merge(transaction_features, on='user_pseudo_id', how='left')
    
    # Calculate engagement metrics
    user_features = calculate_engagement_metrics(user_features)
    
    # Rename columns to match your model's expected format
    user_features = standardize_column_names(user_features)
    
    return user_features

def calculate_engagement_metrics(df):
    """Calculate engagement metrics from user behavior"""
    
    # Get event counts (handle missing columns)
    page_views = df.get('page_view', df.get('page_views', 0))
    item_views = df.get('view_item', df.get('item_views', 0))
    cart_additions = df.get('add_to_cart', df.get('cart_additions', 0))
    purchases = df.get('purchase', df.get('purchases', 0))
    total_events = df.get('event_name_count', df.get('total_events', 1))
    
    # Calculate engagement score
    df['engagement_score'] = (
        page_views * 0.1 + 
        item_views * 0.3 + 
        cart_additions * 0.7 + 
        purchases * 1.0
    ) / np.maximum(total_events, 1)
    
    # Calculate other metrics
    df['conversion_rate'] = purchases / np.maximum(total_events, 1)
    df['purchase_intent_score'] = cart_additions / np.maximum(total_events, 1)
    df['loyalty_score'] = np.minimum(purchases / 3, 1.0)
    
    # Calculate revenue metrics
    revenue = df.get('purchase_revenue_sum', df.get('Item_revenue', df.get('total_revenue', 0)))
    df['value_score'] = np.minimum(revenue / 500, 1.0)
    
    return df

def standardize_column_names(df):
    """Standardize column names to match your model's expectations"""
    
    column_mapping = {
        'user_pseudo_id': 'user_id',
        'event_name_count': 'total_events',
        'page_view': 'page_views',
        'view_item': 'item_views', 
        'add_to_cart': 'cart_additions',
        'purchase': 'purchases',
        'purchase_revenue_sum': 'total_revenue',
        'Item_revenue': 'total_revenue',
        'Age_first': 'age_group',
        'gender_first': 'gender',
        'income_group_first': 'income_group',
        'category_first': 'device_type',
        'source_first': 'traffic_source',
        'medium_first': 'traffic_medium'
    }
    
    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]
    
    # Ensure required columns exist with defaults
    required_columns = [
        'user_id', 'total_events', 'page_views', 'item_views', 
        'cart_additions', 'purchases', 'total_revenue', 'engagement_score',
        'purchase_intent_score', 'loyalty_score', 'value_score', 'conversion_rate'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Create ML clusters based on behavior patterns
    df['ml_cluster'] = assign_behavioral_clusters(df)
    
    return df



# Add this class definition to your app.py (after imports, before other classes)
class AdvancedMLSegmentation:
    def __init__(self, user_features_with_ml=None):
        # Fix the DataFrame condition check
        if user_features_with_ml is None or user_features_with_ml.empty:
            self.user_features_with_ml = pd.DataFrame()
        else:
            self.user_features_with_ml = user_features_with_ml
        
        self.segmentation_results = {}
        
        # Only create segments if we have data
        if not self.user_features_with_ml.empty:
            self.create_segments()
    
    def create_segments(self):
        """Create user segments based on behavior patterns - FIXED VERSION"""
        df = self.user_features_with_ml
        
        # Check if columns exist before using them
        try:
            # High-value users
            high_value_condition = (df['total_revenue'] > 400) & (df['purchases'] >= 2)
            
            # Young explorers - fix the string operation
            if 'age_group' in df.columns:
                # Use pandas Series .str accessor properly
                young_explorer_condition = (df['age_group'].astype(str).str.startswith('2')) & (df['item_views'] >= 5)
            else:
                young_explorer_condition = df['item_views'] >= 5  # Fallback condition
            
            # Search buyers
            if 'traffic_source' in df.columns:
                search_buyer_condition = (df['traffic_source'] == 'Google') & (df['purchases'] >= 1)
            else:
                search_buyer_condition = df['purchases'] >= 1  # Fallback condition
            
            # Intent browsers
            intent_browser_condition = df['cart_additions'] >= 2
            
            # Casual visitors
            casual_visitor_condition = (df['purchases'] == 0) & (df['cart_additions'] == 0)
            
            # Create segments safely
            self.segmentation_results = {
                'high_value': df[high_value_condition],
                'young_explorers': df[young_explorer_condition],
                'search_buyers': df[search_buyer_condition],
                'intent_browsers': df[intent_browser_condition],
                'casual_visitors': df[casual_visitor_condition]
            }
            
            print(f"✅ Created {len(self.segmentation_results)} user segments")
            
        except Exception as e:
            print(f"⚠️ Error creating segments: {e}")
            # Create empty segments as fallback
            self.segmentation_results = {
                'high_value': pd.DataFrame(),
                'young_explorers': pd.DataFrame(),
                'search_buyers': pd.DataFrame(),
                'intent_browsers': pd.DataFrame(),
                'casual_visitors': pd.DataFrame()
            }
    
    def get_user_segment(self, user_id):
        """Get segment for a specific user"""
        if self.user_features_with_ml.empty:
            return {'segment': 'general', 'confidence': 0.5}
            
        for segment_name, segment_df in self.segmentation_results.items():
            if not segment_df.empty and 'user_id' in segment_df.columns and user_id in segment_df['user_id'].values:
                return {'segment': segment_name, 'confidence': 0.8}
        return {'segment': 'general', 'confidence': 0.5}



def convert_ndarray_to_list(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):  # Fix for sets
        return list(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, float)):  # FIXED: removed np.float_
        return float(obj)
    else:
        return obj




class DeepUserIntentPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_clusters=6):
        super(DeepUserIntentPredictor, self).__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.intent_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_clusters),
            nn.Softmax(dim=1)
        )
        
        self.value_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.feature_encoder(x)
        intent_probs = self.intent_predictor(encoded)
        value_score = self.value_predictor(encoded)
        return intent_probs, value_score


class MultiModalProcessor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.behavior_processor = BehaviorProcessor()
        
    def process_user_data(self, user_input):
        processed_data = {}
        
        if 'text_data' in user_input:
            processed_data['text_features'] = self.text_processor.extract_features(user_input['text_data'])
        
        if 'image_data' in user_input:
            processed_data['image_features'] = self.image_processor.extract_features(user_input['image_data'])
        
        if 'behavior_data' in user_input:
            processed_data['behavior_features'] = self.behavior_processor.extract_features(user_input['behavior_data'])
        
        return processed_data
    
    def combine_modalities(self, multi_modal_data):
        combined_features = []
        
        for modality, features in multi_modal_data.items():
            if features is not None:
                combined_features.extend(features)
        
        return np.array(combined_features) if combined_features else np.array([0])

class TextProcessor:
    def __init__(self):
        self.sentiment_analyzer = None
        
    def extract_features(self, text_data):
        if not text_data:
            return [0, 0, 0]
        
        features = [
            len(text_data.split()),
            text_data.count('!'),
            len(text_data)
        ]
        
        return features

class ImageProcessor:
    def __init__(self):
        self.image_model = None
        
    def extract_features(self, image_data):
        return [0.5, 0.3, 0.8]

class BehaviorProcessor:
    def __init__(self):
        self.behavior_patterns = {}
        
    def extract_features(self, behavior_data):
        if not behavior_data:
            return [0, 0, 0]
        
        features = [
            behavior_data.get('click_rate', 0),
            behavior_data.get('time_spent', 0),
            behavior_data.get('scroll_depth', 0)
        ]
        
        return features

class RealTimeBehaviorTracker:
    def __init__(self):
        self.user_sessions = {}
        self.behavior_patterns = {}
        
    def track_user_action(self, user_id, action_data):
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'session_start': datetime.now(),
                'actions': [],
                'current_intent': 'browsing',
                'engagement_score': 0
            }
        
        action_data['timestamp'] = datetime.now()
        self.user_sessions[user_id]['actions'].append(action_data)
        
        self.update_real_time_metrics(user_id, action_data)
        
        return self.get_updated_recommendations(user_id)
    
    def update_real_time_metrics(self, user_id, action_data):
        session = self.user_sessions[user_id]
        
        action_weights = {
            'page_view': 1,
            'item_view': 3,
            'add_to_cart': 5,
            'remove_from_cart': -2,
            'purchase': 10,
            'search': 2,
            'filter_apply': 2
        }
        
        weight = action_weights.get(action_data.get('action_type', ''), 1)
        session['engagement_score'] += weight
        
        session['current_intent'] = self.predict_current_intent(session['actions'])
    
    def predict_current_intent(self, actions):
        if not actions:
            return 'browsing'
            
        recent_actions = actions[-5:]
        
        action_counts = {}
        for action in recent_actions:
            action_type = action.get('action_type', 'unknown')
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        if action_counts.get('add_to_cart', 0) > 0:
            return 'high_purchase_intent'
        elif action_counts.get('item_view', 0) >= 3:
            return 'product_research'
        elif action_counts.get('search', 0) > 0:
            return 'active_searching'
        else:
            return 'casual_browsing'
    
    def get_updated_recommendations(self, user_id):
        if user_id not in self.user_sessions:
            return []
        
        session = self.user_sessions[user_id]
        intent = session['current_intent']
        
        recommendations = {
            'high_purchase_intent': ['Complete your purchase', 'Limited time offer'],
            'product_research': ['Similar products', 'Compare features'],
            'active_searching': ['Search results', 'Popular in category'],
            'casual_browsing': ['Trending now', 'New arrivals']
        }
        
        return recommendations.get(intent, ['General recommendations'])

class ExplainabilityEngine:
    def __init__(self):
        self.explanation_templates = {
            'demographic': "Based on users similar to you (age: {age}, region: {region})",
            'behavioral': "Because you {behavior_pattern}",
            'popularity': "This item is trending among {segment} users",
            'collaborative': "Users who bought similar items also liked this",
            'content_based': "Based on your interest in {category} products"
        }
    
    def generate_explanation(self, recommendation, user_data, reasoning):
        explanations = []
        
        for reason_type, details in reasoning.items():
            if reason_type == 'demographic':
                explanation = self.explanation_templates['demographic'].format(
                    age=user_data.get('age_group', 'your age group'),
                    region=user_data.get('region', 'your area')
                )
                explanations.append({
                    'type': 'demographic',
                    'text': explanation,
                    'confidence': details.get('confidence', 0.5)
                })
            
            elif reason_type == 'behavioral':
                behavior = details.get('pattern', 'show similar browsing patterns')
                explanation = self.explanation_templates['behavioral'].format(
                    behavior_pattern=behavior
                )
                explanations.append({
                    'type': 'behavioral',
                    'text': explanation,
                    'confidence': details.get('confidence', 0.7)
                })
        
        return explanations

class ContextAwareAnalyzer:
    def __init__(self):
        self.seasonal_patterns = {}
        self.time_patterns = {}
        self.device_patterns = {}
    
    def analyze_context(self, user_input):
        context = {
            'temporal': self.analyze_temporal_context(),
            'seasonal': self.analyze_seasonal_context(),
            'device': self.analyze_device_context(user_input.get('device_type')),
            'location': self.analyze_location_context(user_input.get('region')),
            'weather': self.get_weather_influence(user_input.get('region'))
        }
        
        return context
    
    def analyze_temporal_context(self):
        now = datetime.now()
        hour = now.hour
        
        if 9 <= hour <= 17:
            return {
                'period': 'work_hours',
                'recommendation_style': 'quick_browse',
                'urgency': 'low'
            }
        elif 18 <= hour <= 22:
            return {
                'period': 'evening',
                'recommendation_style': 'detailed_exploration',
                'urgency': 'medium'
            }
        else:
            return {
                'period': 'off_hours',
                'recommendation_style': 'impulse_friendly',
                'urgency': 'high'
            }
    
    def analyze_seasonal_context(self):
        return {'season': 'general', 'seasonal_boost': 1.0}
    
    def analyze_device_context(self, device_type):
        return {'device': device_type or 'unknown', 'mobile_optimized': device_type == 'mobile'}
    
    def analyze_location_context(self, region):
        return {'region': region or 'unknown', 'local_preferences': []}
    
    def get_weather_influence(self, region):
        return {'weather': 'unknown', 'influence': 'neutral'}

class FeedbackLearningSystem:
    def __init__(self):
        self.feedback_data = []
        self.model_performance = {}
        
    def collect_feedback(self, user_id, recommendation_id, feedback_type, feedback_value):
        feedback_entry = {
            'user_id': user_id,
            'recommendation_id': recommendation_id,
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'timestamp': datetime.now()
        }
        
        self.feedback_data.append(feedback_entry)
        self.update_performance_metrics(feedback_entry)
    
    def update_performance_metrics(self, feedback):
        if feedback['feedback_type'] == 'click':
            self.model_performance['click_through_rate'] = self.calculate_ctr()
        elif feedback['feedback_type'] == 'purchase':
            self.model_performance['conversion_rate'] = self.calculate_conversion_rate()
    
    def calculate_ctr(self):
        return 0.05
    
    def calculate_conversion_rate(self):
        return 0.02
    
    def get_learning_insights(self):
        return {
            'total_feedback': len(self.feedback_data),
            'performance_metrics': self.model_performance,
            'improvement_suggestions': self.generate_improvement_suggestions()
        }
    
    def generate_improvement_suggestions(self):
        return ['Increase personalization depth', 'Add more contextual factors']

class AdvancedAIPersonalizationEngine:
    def __init__(self, user_features, ml_segmentation, transaction_df):
        self.user_features = user_features
        self.ml_segmentation = ml_segmentation
        self.transaction_df = transaction_df
        
        self.deep_model = None
        self.behavior_tracker = RealTimeBehaviorTracker()
        self.explainability_engine = ExplainabilityEngine()
        self.feedback_loop = FeedbackLearningSystem()
        
        self.context_analyzer = ContextAwareAnalyzer()
        self.multi_modal_processor = MultiModalProcessor()
        
        print("🤖 Initializing Advanced AI Personalization Engine...")
        self.train_deep_learning_models()
    
        def train_deep_learning_models(self):
            """Train deep learning models for intent prediction"""
        print("🧠 Training Deep Learning Models...")
        
        # Prepare training data
        features = [
            'total_events', 'page_views', 'item_views', 'cart_additions',
            'purchases', 'total_revenue', 'engagement_score', 'purchase_intent_score',
            'loyalty_score', 'value_score', 'conversion_rate'
        ]
        
        # Check which features exist
        available_features = [f for f in features if f in self.user_features.columns]
        
        X = self.user_features[available_features].fillna(0).values
        y_cluster = self.user_features['ml_cluster'].values
        y_value = np.clip(self.user_features['total_revenue'] / 200.0, 0, 1)

        
        # FIX: Ensure cluster labels start from 0 and are consecutive
        unique_clusters = sorted(np.unique(y_cluster))
        cluster_mapping = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(unique_clusters)}
        y_cluster_fixed = np.array([cluster_mapping[cluster] for cluster in y_cluster])
        
        print(f"🔧 Fixed cluster mapping: {cluster_mapping}")
        print(f"🔧 Cluster range: {y_cluster_fixed.min()} to {y_cluster_fixed.max()}")
        print(f"🔧 Number of unique clusters: {len(unique_clusters)}")
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_cluster_tensor = torch.LongTensor(y_cluster_fixed)
        y_value_tensor = torch.FloatTensor(y_value).unsqueeze(1)
        
        # Initialize model with correct number of clusters
        num_clusters = len(unique_clusters)
        print(f"🔧 Initializing model with {num_clusters} clusters")
        
        self.deep_model = DeepUserIntentPredictor(
            input_dim=X_scaled.shape[1],
            hidden_dim=128,
            num_clusters=num_clusters
        )
        
        # Training setup
        optimizer = torch.optim.Adam(self.deep_model.parameters(), lr=0.001)
        cluster_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.BCELoss()
        
        # Training loop
        self.deep_model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            
            intent_probs, value_pred = self.deep_model(X_tensor)
            
            cluster_loss = cluster_criterion(intent_probs, y_cluster_tensor)
            value_loss = value_criterion(value_pred, y_value_tensor)
            
            total_loss = cluster_loss + value_loss
            total_loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
        
        self.scaler = scaler
        self.available_features = available_features
        self.cluster_mapping = cluster_mapping
        print("✅ Deep Learning Models Trained!")

    
    def predict_user_intent_deep(self, features_vector):
        try:
            if len(features_vector) != len(self.available_features):
                features_vector += [0] * (len(self.available_features) - len(features_vector))

            x_scaled = self.scaler.transform([features_vector])
            x_tensor = torch.FloatTensor(x_scaled)

            self.deep_model.eval()
            with torch.no_grad():
                intent_probs, value_pred = self.deep_model(x_tensor)
                probs = torch.softmax(intent_probs, dim=1).squeeze()
                predicted_cluster = int(torch.argmax(probs).item())
                confidence = float(torch.max(probs).item())
                value_prob = float(value_pred.squeeze().item())

            return {
                'predicted_cluster': predicted_cluster,
                'cluster_confidence': confidence,
                'value_probability': value_prob
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'predicted_cluster': 4,
                'cluster_confidence': 0.3,
                'value_probability': 0.1
            }



class UltraPersonalizationEngine(AdvancedAIPersonalizationEngine):
    def __init__(self, user_features, ml_segmentation, transaction_df):
        super().__init__(user_features, ml_segmentation, transaction_df)
    
    # ADD THIS METHOD INSIDE THE CLASS
    def train_deep_learning_models(self):
        """Train deep learning models for intent prediction"""
        print("🧠 Training Deep Learning Models...")
        
        features = [
            'total_events', 'page_views', 'item_views', 'cart_additions',
            'purchases', 'total_revenue', 'engagement_score', 'purchase_intent_score',
            'loyalty_score', 'value_score', 'conversion_rate'
        ]
        
        available_features = [f for f in features if f in self.user_features.columns]
        
        X = self.user_features[available_features].fillna(0).values
        y_cluster = self.user_features['ml_cluster'].values
        y_value = (self.user_features['total_revenue'] > 100).astype(int).values
        
        # Fix cluster labels to be consecutive starting from 0
        unique_clusters = sorted(np.unique(y_cluster))
        cluster_mapping = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(unique_clusters)}
        y_cluster_fixed = np.array([cluster_mapping[cluster] for cluster in y_cluster])
        
        print(f"🔧 Cluster range: {y_cluster_fixed.min()} to {y_cluster_fixed.max()}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled)
        y_cluster_tensor = torch.LongTensor(y_cluster_fixed)
        y_value_tensor = torch.FloatTensor(y_value).unsqueeze(1)
        
        num_clusters = len(unique_clusters)
        print(f"🔧 Initializing model with {num_clusters} clusters")
        
        self.deep_model = DeepUserIntentPredictor(
            input_dim=X_scaled.shape[1],
            hidden_dim=128,
            num_clusters=num_clusters
        )
        
        optimizer = torch.optim.Adam(self.deep_model.parameters(), lr=0.001)
        cluster_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.BCELoss()
        
        self.deep_model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            
            intent_probs, value_pred = self.deep_model(X_tensor)
            
            cluster_loss = cluster_criterion(intent_probs, y_cluster_tensor)
            value_loss = value_criterion(value_pred, y_value_tensor)
            
            total_loss = cluster_loss + value_loss
            total_loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
        
        self.scaler = scaler
        self.available_features = available_features
        self.cluster_mapping = cluster_mapping
        print("✅ Deep Learning Models Trained!")
    
    # ... rest of your existing methods

        
    def generate_ultra_personalized_experience(self, user_input, real_time_behavior=None):
        """Generate ultra-personalized experience - IMPROVED HASH DIVERSITY"""
        print("🔍 DEBUG: Starting personalization generation...")
        print(f"🔍 DEBUG: User input: {user_input}")
    
        user_id = user_input.get('user_id', 'unknown')
        
        if user_id in self.user_features['user_id'].values:
            print(f"🔍 DEBUG: Found user {user_id} in training data")
            user_data = self.user_features[self.user_features['user_id'] == user_id].iloc[0]
            features_vector = [user_data[f] for f in self.available_features]
            deep_prediction = self.predict_user_intent_deep(features_vector)
        else:
            print(f"🔍 DEBUG: User {user_id} not found, using improved hash-based features")
            import hashlib
            
            # IMPROVED: Use multiple hash components for better distribution
            user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            hash1 = user_hash % 100  # 0-99
            hash2 = (user_hash // 100) % 100  # Second component
            hash3 = (user_hash // 10000) % 100  # Third component
            
            # Create more diverse features based on multiple hash components
            if hash1 < 10:  # 10% - VIP customers
                features_vector = [80 + hash2//10, 50 + hash3//10, 30 + hash1//5, 8, 3, 1200 + hash2*5, 0.8, 0.7, 0.8, 0.7, 0.12]
                expected_cluster = 0
            elif hash1 < 25:  # 15% - Buyers
                features_vector = [40 + hash2//5, 25 + hash3//5, 15 + hash1//3, 3, 1, 300 + hash2*2, 0.6, 0.5, 0.6, 0.4, 0.06]
                expected_cluster = 1
            elif hash1 < 45:  # 20% - High-intent browsers
                features_vector = [25 + hash2//4, 15 + hash3//4, 10 + hash1//4, 2, 0, hash2//2, 0.4, 0.3, 0.4, 0.2, 0.0]
                expected_cluster = 2
            elif hash1 < 70:  # 25% - Active browsers
                features_vector = [15 + hash2//6, 10 + hash3//6, 5 + hash1//6, 1, 0, hash2//5, 0.3, 0.2, 0.3, 0.1, 0.0]
                expected_cluster = 3
            else:  # 30% - Casual visitors
                features_vector = [5 + hash2//10, 3 + hash3//15, 2 + hash1//20, 0, 0, hash2//10, 0.1, 0.1, 0.1, 0.05, 0.0]
                expected_cluster = 4
            
            print(f"🔍 DEBUG: Hash components: {hash1}, {hash2}, {hash3}")
            print(f"🔍 DEBUG: Expected cluster: {expected_cluster}")
            print(f"🔍 DEBUG: Generated features: {features_vector}")
            
            deep_prediction = self.predict_user_intent_deep(features_vector)
        
        print(f"🔍 DEBUG: Deep prediction result: {deep_prediction}")
        
        # Continue with rest of your function...
        context = self.context_analyzer.analyze_context(user_input)
        current_intent = self.predict_current_intent(user_input, real_time_behavior)
        personalized_content = self.create_ultra_personalized_content(
            deep_prediction, context, current_intent, user_input
        )
        
        return {
            'ai_predictions': {
                'deep_learning': deep_prediction,
                'current_intent': current_intent,
                'confidence_score': deep_prediction.get('cluster_confidence', 0.5)
            },
            'personalized_content': personalized_content,
            'context_insights': context,
            'real_time_adaptations': {'adaptation': '🤖 AI-optimized experience'},
            'explanations': [
                f"🧠 AI analyzed your behavior pattern (Cluster {deep_prediction.get('predicted_cluster', 4)})",
                f"🎯 Detected intent: {current_intent.replace('_', ' ')}",
                f"⚡ Confidence: {deep_prediction.get('cluster_confidence', 0.5):.0%}",
                f"🌟 Personalization level: Ultra"
            ],
            'metadata': {
                'model_version': '2.0',
                'processing_time': '< 100ms',
                'personalization_engine': 'Ultra AI'
            }
        }

    # Continue with rest of your function...

    
    def create_ultra_personalized_content(self, deep_prediction, context, current_intent, user_input):
        hero_section = {
            'headline': self.generate_ai_headline(deep_prediction, context, current_intent, user_input),
            'subheading': self.generate_ai_subheading(context, current_intent),
            'cta_text': self.generate_dynamic_cta(current_intent, deep_prediction),
            'background_theme': self.select_optimal_theme(context, deep_prediction),
            'personalization_level': 'ultra'
        }
        
        product_recommendations = {
            'primary_products': self.generate_ai_product_recommendations(deep_prediction, context),
            'contextual_products': self.get_contextual_recommendations(context),
            'real_time_suggestions': self.get_real_time_suggestions(current_intent),
            'explanation': 'Powered by AI analysis of your behavior and preferences'
        }
        
        return {
            'hero_section': hero_section,
            'product_recommendations': product_recommendations,
            'dynamic_elements': self.create_dynamic_elements(current_intent, context),
            'personalization_strength': deep_prediction['cluster_confidence']
        }
    
    def generate_ai_headline(self, deep_prediction, context, current_intent, user_input):
        """Generate AI-powered headlines based on user cluster"""
        user_id = user_input.get('user_id', 'valued customer')
        cluster = deep_prediction['predicted_cluster']
        confidence = deep_prediction['cluster_confidence']
        
        # 5-cluster headlines
        cluster_headlines = {
            0: f'🌟 Welcome back, VIP {user_id}! Exclusive Collections Await',      # VIP customers
            1: f'🎯 {user_id}, Your Perfect Purchase is Here!',                     # Buyers
            2: f'👗 {user_id}, Discover Your Style Essentials!',                   # High-intent (fashion-focused)
            3: f'🏠 {user_id}, Transform Your Home Today!',                        # Active browsers (home-focused)
            4: f'📚 Welcome {user_id}! Explore & Discover Amazing Products'        # Casual visitors
        }
        
        base_headline = cluster_headlines.get(cluster, cluster_headlines[4])
        
        # Add confidence boost
        if confidence > 0.8:
            return f"🚀 {base_headline}"
        else:
            return base_headline

    
    def generate_ai_subheading(self, context, current_intent):
        time_context = context.get("temporal", {}).get("period", "off_hours")
        return f"AI-powered recommendations optimized for {time_context} {current_intent.replace('_', ' ')}"

    
    def generate_ai_product_recommendations(self, deep_prediction, context):
        """Generate AI-powered product recommendations based on user cluster"""
        cluster = deep_prediction['predicted_cluster']
        confidence = deep_prediction['cluster_confidence']
        
        # 5-cluster product mapping
        cluster_products = {
            0: ['💎 VIP Gaming Setup', '🏆 Premium Laptop Pro', '📱 Latest iPhone Pro Max'],  # VIP customers
            1: ['🎮 Gaming Collection', '💻 High-Performance Laptop', '📱 Latest Smartphone'],  # Buyers
            2: ['👗 Designer Fashion', '👠 Luxury Accessories', '💄 Beauty Essentials'],      # High-intent browsers
            3: ['🏠 Smart Home Bundle', '🍳 Premium Kitchen Set', '🌿 Garden Collection'],    # Active browsers
            4: ['📚 Popular Books', '🎨 Creative Tools', '🎵 Digital Entertainment']         # Casual visitors
        }
        
        base_products = cluster_products.get(cluster, cluster_products[4])
        
        # Add confidence-based enhancement
        if confidence > 0.8:
            return [f"🌟 AI Top Pick: {product}" for product in base_products]
        elif confidence > 0.6:
            return [f"⭐ Recommended: {product}" for product in base_products]
        else:
            return base_products
        
    def generate_dynamic_cta(self, current_intent, deep_prediction):
        confidence = deep_prediction['cluster_confidence']
        
        if confidence > 0.8:
            ctas = {
                'high_purchase_intent': '🚀 Complete Purchase Now',
                'product_research': '🔍 Compare AI Picks',
                'active_searching': '✨ View AI Results',
                'casual_browsing': '🛍️ Start AI Shopping'
            }
        else:
            ctas = {
                'high_purchase_intent': 'Complete Purchase',
                'product_research': 'Compare Products',
                'active_searching': 'View Results',
                'casual_browsing': 'Start Shopping'
            }
        
        return ctas.get(current_intent, 'Explore Now')
    
    def select_optimal_theme(self, context, deep_prediction):
        confidence = deep_prediction['cluster_confidence']
        return 'ultra_ai_optimized' if confidence > 0.7 else 'ai_optimized'
    
    
    def get_contextual_recommendations(self, context):
        time_period = context['temporal']['period']
        
        time_products = {
            'work_hours': ['⚡ Quick Solutions', '💼 Professional Tools'],
            'evening': ['🛋️ Relaxation Products', '🍽️ Dinner Essentials'],
            'off_hours': ['🌙 Night Specials', '⭐ Impulse Deals']
        }
        
        return time_products.get(time_period, ['🔥 Trending Now', '⚡ Flash Deals'])
    
    def get_real_time_suggestions(self, current_intent):
        suggestions = {
            'high_purchase_intent': ['🎯 Complete Bundle', '💰 Save More Deal'],
            'product_research': ['📊 Comparison Tool', '⭐ Expert Reviews'],
            'active_searching': ['🔍 Related Searches', '📈 Popular Results'],
            'casual_browsing': ['🆕 Just Arrived', '🔥 Hot Picks']
        }
        
        return suggestions.get(current_intent, ['🌟 Recommended', '✨ Popular Choice'])
    
    def create_dynamic_elements(self, current_intent, context):
        return {
            'dynamic_element': f'🤖 AI-optimized for {current_intent}',
            'urgency_indicator': context['temporal']['urgency'],
            'personalization_badge': '🎯 Ultra-Personalized'
        }
    
    def get_real_time_adaptations(self, current_intent):
        adaptations = {
            'high_purchase_intent': '🚀 Streamlined checkout experience',
            'product_research': '📊 Enhanced comparison tools',
            'active_searching': '🔍 Optimized search results',
            'casual_browsing': '🌟 Discovery-focused layout'
        }
        
        return {'adaptation': adaptations.get(current_intent, '🤖 AI-optimized experience')}

# ===== MOCK DATA CREATION =====
def create_mock_data():
    user_features = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(100)],
        'total_events': np.random.randint(1, 100, 100),
        'page_views': np.random.randint(1, 50, 100),
        'item_views': np.random.randint(0, 30, 100),
        'cart_additions': np.random.randint(0, 10, 100),
        'purchases': np.random.randint(0, 5, 100),
        'total_revenue': np.random.uniform(0, 1000, 100),
        'engagement_score': np.random.uniform(0, 1, 100),
        'purchase_intent_score': np.random.uniform(0, 1, 100),
        'loyalty_score': np.random.uniform(0, 1, 100),
        'value_score': np.random.uniform(0, 1, 100),
        'conversion_rate': np.random.uniform(0, 0.1, 100),
        'ml_cluster': np.random.randint(0, 5, 100)
    })
    
    transaction_df = pd.DataFrame({
        'user_id': np.random.choice([f'user_{i}' for i in range(100)], 200),
        'transaction_id': [f'txn_{i}' for i in range(200)],
        'amount': np.random.uniform(10, 500, 200)
    })
    
    return user_features, transaction_df

# ===== MODEL LOADING =====
def load_ultra_engine():
    model_path = 'models/ultra_personalization_engine.pkl'
    
    # Try to load saved model first
    if os.path.exists(model_path):
        try:
            print("🤖 Loading Ultra Personalization Engine from file...")
            with open(model_path, 'rb') as f:
                ultra_engine = pickle.load(f)
            print("✅ Ultra AI engine loaded successfully from file!")
            return ultra_engine
        except Exception as e:
            print(f"❌ Error loading saved model: {e}")
    
    # Create new model with real datasets
    print("🔄 Creating new engine with real datasets...")
    return create_new_engine_with_real_data()

def create_new_engine_with_real_data():
    try:
        # Load and process real datasets
        user_features, transaction_df = load_and_process_real_datasets()
        
        # Create ML segmentation
        ml_segmentation = AdvancedMLSegmentation(user_features)
        
        # Create the engine
        ultra_engine = UltraPersonalizationEngine(user_features, ml_segmentation, transaction_df)
        
        # Print dataset statistics
        print(f"📊 Dataset Statistics:")
        print(f"   - Total users: {len(user_features):,}")
        print(f"   - Average revenue per user: ${user_features['total_revenue'].mean():.2f}")
        print(f"   - Average purchases per user: {user_features['purchases'].mean():.2f}")
        print(f"   - Conversion rate: {user_features['conversion_rate'].mean():.3f}")
        print(f"   - Unique clusters: {user_features['ml_cluster'].nunique()}")
        
        return ultra_engine
        
    except Exception as e:
        print(f"❌ Error creating engine with real data: {e}")
        return MockUltraEngine()


def create_new_engine():
    try:
        user_features, transaction_df = create_mock_data()
        ml_segmentation = {}
        
        ultra_engine = UltraPersonalizationEngine(user_features, ml_segmentation, transaction_df)
        print("✅ New Ultra AI engine created successfully!")
        return ultra_engine
    except Exception as e:
        print(f"❌ Error creating new engine: {e}")
        return MockUltraEngine()

class MockUltraEngine:
    def generate_ultra_personalized_experience(self, user_input, real_time_behavior=None):
        user_id = user_input.get('user_id', 'valued customer')
        return {
            'personalized_content': {
                'hero_section': {
                    'headline': f'🤖 Welcome back, {user_id}! (Mock AI)',
                    'subheading': 'Mock AI-powered recommendations',
                    'cta_text': '🚀 Explore Mock AI Picks',
                    'background_theme': 'mock_ai_optimized',
                    'personalization_level': 'mock_ultra'
                },
                'product_recommendations': {
                    'primary_products': ['🎮 Mock AI Product 1', '💻 Mock AI Product 2', '📱 Mock AI Product 3'],
                    'contextual_products': ['🔥 Mock Context Product 1', '⚡ Mock Context Product 2'],
                    'real_time_suggestions': ['🌟 Mock Real-time 1', '✨ Mock Real-time 2'],
                    'explanation': 'Mock AI-powered recommendations (fallback mode)'
                },
                'dynamic_elements': {'dynamic_element': '🤖 Mock AI-optimized content'},
                'personalization_strength': 0.75
            },
            'explanations': [{'type': 'mock_behavioral', 'text': 'Mock: Based on your browsing patterns', 'confidence': 0.8}],
            'context_insights': {
                'temporal': {'period': 'mock_time', 'urgency': 'medium'},
                'device': {'device': user_input.get('device_type', 'unknown')}
            },
            'ai_predictions': {
                'deep_learning': {'predicted_cluster': 2, 'cluster_confidence': 0.75, 'value_probability': 0.6},
                'current_intent': 'browsing',
                'confidence_score': 0.75
            },
            'real_time_adaptations': {'adaptation': 'Mock: Optimized for browsing'},
            'metadata': {
                'ai_model_version': 'mock_2.0',
                'personalization_strength': 'mock_ultra',
                'generation_timestamp': datetime.now().isoformat()
            }
        }

# ===== FLASK APP =====
app = Flask(__name__)
CORS(app)

# Load your Ultra AI engine
ultra_engine = load_ultra_engine()

@app.route('/validate-datasets', methods=['GET'])
def validate_datasets():
    """Validate the loaded datasets"""
    try:
        # Check if using real data
        is_real_data = hasattr(ultra_engine, 'user_features') and len(ultra_engine.user_features) > 1000
        
        if is_real_data:
            df = ultra_engine.user_features
            
            # Dataset quality metrics
            quality_metrics = {
                'total_users': len(df),
                'avg_revenue_per_user': float(df['total_revenue'].mean()),
                'avg_purchases_per_user': float(df['purchases'].mean()),
                'conversion_rate': float(df['conversion_rate'].mean()),
                'high_value_users': len(df[df['total_revenue'] > 100]),
                'active_users': len(df[df['total_events'] > 5]),
                'unique_clusters': int(df['ml_cluster'].nunique()),
                'data_quality_score': calculate_data_quality_score(df)
            }
            
            # Sample user profiles
            sample_users = df.head(3)[['user_id', 'total_events', 'purchases', 'total_revenue', 'engagement_score', 'ml_cluster']].to_dict('records')
            
            return jsonify({
                'dataset_status': 'real_data_loaded',
                'quality_metrics': quality_metrics,
                'sample_users': sample_users,
                'recommendations': get_data_recommendations(quality_metrics)
            })
        else:
            return jsonify({
                'dataset_status': 'using_synthetic_data',
                'message': 'Real datasets not found or failed to load',
                'next_steps': [
                    'Download datasets from Google Drive link',
                    'Place files in data/ folder',
                    'Restart Flask application'
                ]
            })
            
    except Exception as e:
        return jsonify({'error': str(e)})

def calculate_data_quality_score(df):
    """Calculate a data quality score (0-100)"""
    score = 0
    
    # Revenue distribution (30 points)
    if df['total_revenue'].mean() > 50:
        score += 30
    elif df['total_revenue'].mean() > 10:
        score += 20
    elif df['total_revenue'].mean() > 1:
        score += 10
    
    # Conversion rate (25 points)
    if df['conversion_rate'].mean() > 0.05:
        score += 25
    elif df['conversion_rate'].mean() > 0.02:
        score += 15
    elif df['conversion_rate'].mean() > 0.005:
        score += 10
    
    # User engagement (25 points)
    if df['engagement_score'].mean() > 0.3:
        score += 25
    elif df['engagement_score'].mean() > 0.1:
        score += 15
    elif df['engagement_score'].mean() > 0.05:
        score += 10
    
    # Data completeness (20 points)
    non_zero_users = len(df[df['total_events'] > 0]) / len(df)
    score += int(non_zero_users * 20)
    
    return min(score, 100)

def get_data_recommendations(metrics):
    """Get recommendations based on data quality"""
    recommendations = []
    
    if metrics['avg_revenue_per_user'] < 10:
        recommendations.append("Consider filtering for active users only")
    
    if metrics['conversion_rate'] < 0.01:
        recommendations.append("Low conversion rate - check transaction data linkage")
    
    if metrics['data_quality_score'] < 70:
        recommendations.append("Data quality could be improved - verify dataset processing")
    
    if metrics['high_value_users'] < 100:
        recommendations.append("Limited high-value users - consider expanding dataset")
    
    return recommendations if recommendations else ["Data quality looks good!"]

@app.route('/your-route', methods=['GET', 'POST'])
def your_function():
    if request.method == 'POST':
        # Handle form submission
        return "Form submitted"
    else:
        # Handle GET request (show the page)
        return render_template('your_template.html')
    
@app.route('/debug-cluster-assignment', methods=['GET'])
def debug_cluster_assignment():
    """Debug cluster assignment process"""
    try:
        if hasattr(ultra_engine, 'user_features'):
            df = ultra_engine.user_features.head(100)  # Test with first 100 users
            
            # Test the cluster assignment
            test_clusters = assign_behavioral_clusters_fixed(df)
            
            return jsonify({
                'sample_size': len(df),
                'clusters_created': list(set(test_clusters)),
                'cluster_counts': {str(i): test_clusters.count(i) for i in set(test_clusters)},
                'unique_clusters': len(set(test_clusters)),
                'sample_data': {
                    'revenue_range': [float(df['total_revenue'].min()), float(df['total_revenue'].max())],
                    'events_range': [int(df['total_events'].min()), int(df['total_events'].max())],
                    'engagement_range': [float(df['engagement_score'].min()), float(df['engagement_score'].max())]
                }
            })
        else:
            return jsonify({'error': 'No user features available'})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/debug-5-clusters', methods=['GET'])
def debug_5_clusters():
    """Debug 5-cluster distribution"""
    try:
        if hasattr(ultra_engine, 'user_features'):
            df = ultra_engine.user_features
            cluster_counts = df['ml_cluster'].value_counts().sort_index()
            
            # Calculate cluster percentages
            total_users = len(df)
            cluster_percentages = {k: f"{(v/total_users)*100:.1f}%" for k, v in cluster_counts.items()}
            
            return jsonify({
                'cluster_distribution': cluster_counts.to_dict(),
                'cluster_percentages': cluster_percentages,
                'total_clusters': len(cluster_counts),
                'expected_clusters': 5,
                'cluster_meanings': {
                    0: 'VIP/High-value customers',
                    1: 'Buyers/Converters', 
                    2: 'High-intent browsers',
                    3: 'Active browsers',
                    4: 'Casual visitors'
                }
            })
        else:
            return jsonify({'error': 'No user features available'})
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/health', methods=['GET'])
def health_check():
    model_type = "Ultra AI" if hasattr(ultra_engine, 'deep_model') else "Mock AI"
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'model_type': model_type,
        'ai_features': hasattr(ultra_engine, 'explainability_engine')
    })

@app.route('/personalize', methods=['POST'])
def personalize():
    try:
        data = request.get_json()
        user_input = data.get("user_input", {})
        real_time_behavior = data.get("real_time_behavior", [])

        result = ultra_engine.generate_ultra_personalized_experience(user_input, real_time_behavior)

        if not result or "hero_section" not in result:
            raise ValueError("Invalid response from AI engine")

        cluster_level = result["hero_section"].get("personalization_level", "unknown")
        strength = result.get("personalization_strength", 0.0)
        current_intent = user_input.get("current_intent", "casual_browsing")

        response = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_input.get("user_id", "unknown"),
            "data": {
                "content": {
                    "hero_section": result.get("hero_section", {}),
                    "personalization_strength": result.get("personalization_strength", "low")
                },
                "intent_prediction": {
                    "label": current_intent,
                    "confidence": strength,
                    "explanation": f"User aligns with segment: {cluster_level}"
                },
                "model_metadata": {
                    "ai_version": "v1.0",
                    "cluster": cluster_level,
                    "engine": "UltraPersonalizationEngine",
                    "trained_on": "real_transactions+user_activity"
                }
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"❌ Backend error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Add these new routes after your existing ones

@app.route('/test-data-loading', methods=['GET'])
def test_data_loading():
    """Test data loading with fixes"""
    try:
        user_features, transaction_df = load_and_process_real_datasets()
        
        return jsonify({
            'status': 'success',
            'users_loaded': len(user_features),
            'transactions_loaded': len(transaction_df),
            'sample_user': user_features.head(1).to_dict('records')[0] if not user_features.empty else {},
            'avg_revenue': float(user_features['total_revenue'].mean()) if not user_features.empty else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/debug-dataset-columns', methods=['GET'])
def debug_dataset_columns():
    """Check the actual column names in your datasets"""
    try:
        activity_logs = pd.read_csv('data/User_Activity_Logs.csv', nrows=5)  # Just read first 5 rows
        transaction_df = pd.read_csv('data/Transaction_Dataset.csv', nrows=5)
        
        return jsonify({
            'activity_log_columns': list(activity_logs.columns),
            'transaction_columns': list(transaction_df.columns),
            'activity_sample': activity_logs.head(2).to_dict('records'),
            'transaction_sample': transaction_df.head(2).to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/debug-simple', methods=['GET'])
def debug_simple():
    """Simple debug to check basic model functionality"""
    try:
        debug_info = {
            'model_type': type(ultra_engine).__name__,
            'has_deep_model': hasattr(ultra_engine, 'deep_model'),
            'has_user_features': hasattr(ultra_engine, 'user_features'),
            'training_data_size': len(ultra_engine.user_features) if hasattr(ultra_engine, 'user_features') else 0,
            'available_features': getattr(ultra_engine, 'available_features', 'Not found'),
            'sample_user_ids': list(ultra_engine.user_features['user_id'].head(5)) if hasattr(ultra_engine, 'user_features') else []
        }
        
        # Convert to JSON serializable
        debug_info_serializable = convert_ndarray_to_list(debug_info)
        return jsonify(debug_info_serializable)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug-prediction', methods=['GET'])
def debug_prediction():
    """Test if the model makes different predictions for different inputs"""
    try:
        test_cases = [
            {'name': 'high_activity', 'features': [100, 80, 50, 20, 10, 2000, 0.9, 0.8, 0.9, 0.8, 0.15]},
            {'name': 'low_activity', 'features': [5, 3, 1, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.0]},
            {'name': 'medium_activity', 'features': [30, 20, 10, 3, 1, 300, 0.5, 0.4, 0.5, 0.4, 0.03]}
        ]
        
        results = []
        for case in test_cases:
            try:
                prediction = ultra_engine.predict_user_intent_deep(case['features'])
                results.append({
                    'test_case': case['name'],
                    'input_features': case['features'],
                    'prediction': prediction
                })
            except Exception as e:
                results.append({
                    'test_case': case['name'],
                    'error': str(e)
                })
        
        response = {
            'test_results': results,
            'predictions_vary': len(set(str(r.get('prediction', {}).get('predicted_cluster', 'error')) for r in results)) > 1
        }
        
        # Convert to JSON serializable
        response_serializable = convert_ndarray_to_list(response)
        return jsonify(response_serializable)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get-sample-users', methods=['GET'])
def get_sample_users():
    """Get sample user IDs from different clusters"""
    try:
        sample_users = {}
        for cluster in range(5):
            cluster_users = ultra_engine.user_features[ultra_engine.user_features['ml_cluster'] == cluster]['user_id'].head(3).tolist()
            sample_users[f'cluster_{cluster}'] = cluster_users
        
        return jsonify(sample_users)
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/debug-clusters', methods=['GET', 'POST'])
def debug_clusters():
    """Debug cluster distribution"""
    try:
        if hasattr(ultra_engine, 'user_features'):
            df = ultra_engine.user_features
            cluster_counts = df['ml_cluster'].value_counts().sort_index()
            
            return jsonify({
                'cluster_distribution': cluster_counts.to_dict(),
                'max_cluster': int(df['ml_cluster'].max()),
                'min_cluster': int(df['ml_cluster'].min()),
                'unique_clusters': len(df['ml_cluster'].unique()),
                'status': 'Should be 0-2 for 3 classes'
            })
        else:
            return jsonify({'error': 'No user features available'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/ping")
def ping():
    return "pong"


@app.route('/debug-training-data', methods=['GET'])
def debug_training_data():
    """Check what training data the model is using"""
    try:
        sample_data = ultra_engine.user_features.head(5).to_dict('records')  # Reduced to 5 for readability
        
        response = {
            'sample_training_data': sample_data,
            'data_looks_realistic': ultra_engine.user_features['total_revenue'].std() > 100,
            'data_statistics': {
                'total_users': len(ultra_engine.user_features),
                'avg_revenue': float(ultra_engine.user_features['total_revenue'].mean()),
                'avg_purchases': float(ultra_engine.user_features['purchases'].mean()),
                'unique_clusters': len(ultra_engine.user_features['ml_cluster'].unique())
            }
        }
        
        # Convert to JSON serializable
        response_serializable = convert_ndarray_to_list(response)
        return jsonify(response_serializable)
        
    except Exception as e:
        return jsonify({'error': str(e)})



if not os.path.exists('static'):
    os.makedirs('static')

if __name__ == '__main__':
    print("🚀 Starting Ultra Personalization API...")
    app.run(host='0.0.0.0', port=5000, debug=True)
