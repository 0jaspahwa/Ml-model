from datetime import datetime

class ContextAwareAnalyzer:
    def analyze_context(self, user_input):
        return {
            'temporal': self.analyze_temporal_context(),
            'seasonal': self.analyze_seasonal_context(),
            'device': self.analyze_device_context(user_input.get('device_type')),
            'location': self.analyze_location_context(user_input.get('region')),
            'weather': {'weather': 'unknown', 'influence': 'neutral'}
        }

    def analyze_temporal_context(self):
        hour = datetime.now().hour
        if 9 <= hour <= 17:
            return {'period': 'work_hours', 'urgency': 'low'}
        elif 18 <= hour <= 22:
            return {'period': 'evening', 'urgency': 'medium'}
        return {'period': 'off_hours', 'urgency': 'high'}

    def analyze_seasonal_context(self):
        return {'season': 'general'}

    def analyze_device_context(self, device_type):
        return {'device': device_type or 'unknown'}

    def analyze_location_context(self, region):
        return {'region': region or 'unknown'}
