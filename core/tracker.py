from datetime import datetime

class RealTimeBehaviorTracker:
    def __init__(self):
        self.user_sessions = {}

    def track_user_action(self, user_id, action_data):
        session = self.user_sessions.setdefault(user_id, {
            'session_start': datetime.now(),
            'actions': [],
            'current_intent': 'browsing',
            'engagement_score': 0
        })
        action_data['timestamp'] = datetime.now()
        session['actions'].append(action_data)
        self.update_metrics(user_id, action_data)
        return self.get_updated_recommendations(user_id)

    def update_metrics(self, user_id, action_data):
        session = self.user_sessions[user_id]
        weights = {
            'page_view': 1, 'item_view': 3, 'add_to_cart': 5,
            'remove_from_cart': -2, 'purchase': 10, 'search': 2, 'filter_apply': 2
        }
        session['engagement_score'] += weights.get(action_data.get('action_type'), 1)
        session['current_intent'] = self.predict_current_intent(session['actions'])

    def predict_current_intent(self, actions):
        if not actions:
            return 'browsing'
        recent = actions[-5:]
        counts = {}
        for a in recent:
            counts[a.get('action_type', 'unknown')] = counts.get(a.get('action_type'), 0) + 1
        if counts.get('add_to_cart', 0) > 0:
            return 'high_purchase_intent'
        if counts.get('item_view', 0) >= 3:
            return 'product_research'
        if counts.get('search', 0) > 0:
            return 'active_searching'
        return 'casual_browsing'

    def get_updated_recommendations(self, user_id):
        intent = self.user_sessions[user_id]['current_intent']
        mapping = {
            'high_purchase_intent': ['Complete your purchase', 'Limited time offer'],
            'product_research': ['Similar products', 'Compare features'],
            'active_searching': ['Search results', 'Popular in category'],
            'casual_browsing': ['Trending now', 'New arrivals']
        }
        return mapping.get(intent, ['General recommendations'])
