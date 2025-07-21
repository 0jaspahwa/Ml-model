from datetime import datetime

class FeedbackLearningSystem:
    def __init__(self):
        self.feedback_data = []
        self.metrics = {}

    def collect_feedback(self, user_id, rec_id, feedback_type, value):
        entry = {
            'user_id': user_id,
            'recommendation_id': rec_id,
            'feedback_type': feedback_type,
            'feedback_value': value,
            'timestamp': datetime.now()
        }
        self.feedback_data.append(entry)
        self.update_metrics(entry)

    def update_metrics(self, feedback):
        if feedback['feedback_type'] == 'click':
            self.metrics['CTR'] = self.calculate_ctr()
        elif feedback['feedback_type'] == 'purchase':
            self.metrics['CR'] = self.calculate_conversion_rate()

    def calculate_ctr(self):
        return 0.05

    def calculate_conversion_rate(self):
        return 0.02

    def get_learning_insights(self):
        return {
            'total_feedback': len(self.feedback_data),
            'metrics': self.metrics,
            'suggestions': ['Use deeper personalization', 'Add more context awareness']
        }
