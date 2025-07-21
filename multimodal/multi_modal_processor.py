import numpy as np

class MultiModalProcessor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.behavior_processor = BehaviorProcessor()

    def process_user_data(self, user_input):
        data = {}
        if 'text_data' in user_input:
            data['text_features'] = self.text_processor.extract_features(user_input['text_data'])
        if 'image_data' in user_input:
            data['image_features'] = self.image_processor.extract_features(user_input['image_data'])
        if 'behavior_data' in user_input:
            data['behavior_features'] = self.behavior_processor.extract_features(user_input['behavior_data'])
        return data

    def combine_modalities(self, data):
        combined = []
        for features in data.values():
            combined.extend(features)
        return np.array(combined) if combined else np.array([0])

class TextProcessor:
    def extract_features(self, text):
        return [len(text.split()), text.count('!'), len(text)]

class ImageProcessor:
    def extract_features(self, _):
        return [0.5, 0.3, 0.8]  # Placeholder

class BehaviorProcessor:
    def extract_features(self, behavior):
        return [
            behavior.get("click_rate", 0),
            behavior.get("time_spent", 0),
            behavior.get("scroll_depth", 0)
        ]
