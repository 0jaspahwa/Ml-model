class ExplainabilityEngine:
    def __init__(self):
        self.templates = {
            'demographic': "Based on users similar to you (age: {age}, region: {region})",
            'behavioral': "Because you {pattern}"
        }

    def generate_explanation(self, recommendation, user_data, reasoning):
        explanations = []
        for reason_type, details in reasoning.items():
            if reason_type == 'demographic':
                exp = self.templates['demographic'].format(
                    age=user_data.get('age_group', 'your age'),
                    region=user_data.get('region', 'your region')
                )
            elif reason_type == 'behavioral':
                exp = self.templates['behavioral'].format(pattern=details.get('pattern', 'browse often'))
            else:
                exp = "AI recommendation based on your profile"
            explanations.append({
                'type': reason_type,
                'text': exp,
                'confidence': details.get('confidence', 0.5)
            })
        return explanations
