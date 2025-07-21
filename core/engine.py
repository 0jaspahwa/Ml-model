import os
import pickle
from core.segmentation import AdvancedMLSegmentation
from processing.data_loader import load_and_process_real_datasets, create_mock_data
from core.tracker import RealTimeBehaviorTracker
from explainability.explanation_engine import ExplainabilityEngine
from feedback.feedback_loop import FeedbackLearningSystem
from context.context_analyzer import ContextAwareAnalyzer
from multimodal.multi_modal_processor import MultiModalProcessor
from core.models import DeepUserIntentPredictor
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class UltraPersonalizationEngine:
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

        self.train_deep_learning_models()

    def train_deep_learning_models(self):
        features = [
            'total_events', 'page_views', 'item_views', 'cart_additions',
            'purchases', 'total_revenue', 'engagement_score',
            'purchase_intent_score', 'loyalty_score', 'value_score', 'conversion_rate'
        ]
        available_features = [f for f in features if f in self.user_features.columns]
        X = self.user_features[available_features].fillna(0).values
        y_cluster = self.user_features['ml_cluster'].values
        y_value = (self.user_features['total_revenue'] > 100).astype(int).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.scaler = scaler
        self.available_features = available_features
        self.deep_model = DeepUserIntentPredictor(input_dim=X_scaled.shape[1], hidden_dim=128, num_clusters=5)
        # Add training loop here if required

    def generate_ultra_personalized_experience(self, user_input, real_time_behavior=None):
        user_id = user_input.get('user_id', 'unknown')

        if user_id in self.user_features['user_id'].values:
            user_row = self.user_features[self.user_features['user_id'] == user_id].iloc[0]
            features_vector = [user_row[f] for f in self.available_features]
        else:
            features_vector, _ = self.fallback_generate_features(user_id)

        prediction = self.predict_user_intent_deep(features_vector)
        context = self.context_analyzer.analyze_context(user_input)
        intent = self.behavior_tracker.predict_current_intent(real_time_behavior or [])

        return {
            'ai_predictions': {
                'deep_learning': prediction,
                'current_intent': intent,
                'confidence_score': prediction.get('cluster_confidence', 0.5)
            },
            'context_insights': context,
            'real_time_adaptations': {'adaptation': 'AI-optimized'},
            'personalized_content': {
                'hero_section': {
                    'headline': self.generate_ai_headline(prediction, context, intent, user_input),
                    'subheading': self.generate_ai_subheading(context, intent),
                    'cta_text': self.generate_dynamic_cta(intent, prediction),
                    'background_theme': self.select_optimal_theme(context, prediction),
                    'personalization_level': 'ultra'
                }
            }
        }

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
            return {
                'predicted_cluster': 4,
                'cluster_confidence': 0.3,
                'value_probability': 0.1
            }
    
    def fallback_generate_features(self, user_id):
        import hashlib
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        hash1 = user_hash % 100
        hash2 = (user_hash // 100) % 100
        hash3 = (user_hash // 10000) % 100

        if hash1 < 10:
            return [80 + hash2//10, 50 + hash3//10, 30 + hash1//5, 8, 3, 1200 + hash2*5, 0.8, 0.7, 0.8, 0.7, 0.12], 0
        elif hash1 < 25:
            return [40 + hash2//5, 25 + hash3//5, 15 + hash1//3, 3, 1, 300 + hash2*2, 0.6, 0.5, 0.6, 0.4, 0.06], 1
        elif hash1 < 45:
            return [25 + hash2//4, 15 + hash3//4, 10 + hash1//4, 2, 0, hash2//2, 0.4, 0.3, 0.4, 0.2, 0.0], 2
        elif hash1 < 70:
            return [15 + hash2//6, 10 + hash3//6, 5 + hash1//6, 1, 0, hash2//5, 0.3, 0.2, 0.3, 0.1, 0.0], 3
        else:
            return [5 + hash2//10, 3 + hash3//15, 2 + hash1//20, 0, 0, hash2//10, 0.1, 0.1, 0.1, 0.05, 0.0], 4
    
    def generate_ai_headline(self, prediction, context, intent, user_input):
        value_prob = prediction.get('value_probability', 0)
        urgency = context.get('temporal', {}).get('urgency', 'low')

        if value_prob > 0.8 and urgency == 'high':
            return "Don't Miss Out – Your Perfect Match Is Here!"
        elif value_prob > 0.6:
            return "Discover Products You'll Love"
        elif intent == 'high_purchase_intent':
            return "Ready to Checkout? We've Got You Covered"
        elif intent == 'active_searching':
            return "Top Picks Just For You – Find Them Now"
        else:
            return "Explore What’s Trending Today"

    def generate_ai_subheading(self, context, intent):
        device = context.get('device', {}).get('device', 'unknown')
        period = context.get('temporal', {}).get('period', 'anytime')

        if intent == 'product_research':
            return f"Compare features and reviews, optimized for your {device}"
        elif period == 'evening':
            return "Unwind with personalized suggestions curated this evening"
        elif device == 'mobile':
            return "Swipe through trending items on the go"
        else:
            return "A tailored experience just for your needs"

    def generate_dynamic_cta(self, intent, prediction):
        confidence = prediction.get('cluster_confidence', 0.5)

        if intent == 'high_purchase_intent' and confidence > 0.7:
            return "Complete Your Purchase Now"
        elif intent == 'active_searching':
            return "See Full Results"
        elif intent == 'product_research':
            return "Compare and Decide"
        elif intent == 'casual_browsing':
            return "Browse Curated Picks"
        return "Explore More"

    def select_optimal_theme(self, context, prediction):
        urgency = context.get('temporal', {}).get('urgency', 'medium')
        cluster = prediction.get('predicted_cluster', 0)

        themes = {
            0: 'premium_dark',
            1: 'trendy_vibrant',
            2: 'minimal_light',
            3: 'casual_neutral',
            4: 'simple_pastel'
        }

        theme = themes.get(cluster, 'default')
        if urgency == 'high':
            return f"{theme}_highlighted"
        return theme


def load_ultra_engine():
    model_path = 'models/ultra_personalization_engine.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    user_features, transaction_df = load_and_process_real_datasets()
    ml_segmentation = AdvancedMLSegmentation(user_features)
    return UltraPersonalizationEngine(user_features, ml_segmentation, transaction_df)
