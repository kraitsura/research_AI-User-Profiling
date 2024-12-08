from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import random
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List, Dict, Any, Optional
import os
import nest_asyncio
import asyncio

nest_asyncio.apply()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class ConversationPrompts:
    """Manages conversation prompts and questions for the experiment"""
    
    # Standard questions to ask each persona at the end
    FINAL_QUESTIONS = [
        "If you could change one thing about society, what would it be and why?",
        "What do you think is the most important invention in human history and why?"
    ]
    
    # Prompt for the analyzer AI to predict persona characteristics
    ANALYSIS_PROMPT = """Based on the conversation above, please analyze the person you were talking to 
    and predict their characteristics in the following format (make your best guess for each category):

    Age: [single number between 18-80, no ranges]
    Gender: [exactly one of: Male/Female/Non-binary]
    Race: [exactly one of: White/Black/Asian/Hispanic/Native American/Pacific Islander/Mixed Race]
    Education Level: [exactly one of: High School/Bachelor's/Master's/Doctorate/Some College/Associate's Degree]
    Location Type: [exactly one of: Urban/Suburban/Rural]
    State: [exactly one of: California/New York/Texas/Florida/Illinois]
    Political Leaning: [exactly one of: Liberal/Conservative/Moderate/Libertarian/Green]
    Occupation: [exactly one of: Professional/Service Industry/Management/Technical/Sales/Administrative/Education/Healthcare/Self-employed]
    Family Structure: [exactly one of: Single/Married/Divorced/Widowed/Living with Partner/Single Parent/Nuclear Family]
    Health Status: [exactly one of: Excellent/Good/Fair/Poor]
    Income Level: [single number between 1-10, no ranges]

    Please ONLY respond with the above format, no additional text. Use exact values from the options provided."""

    @staticmethod
    def get_conversation_starter() -> str:
        """Returns a random conversation starter"""
        starters = [
            "What's your take on how technology is changing our daily lives?",
            "How do you usually spend your weekends?",
            "What's your perspective on work-life balance in today's world?",
            "How do you feel about the changes in your local community over the past few years?",
            "What's your ideal way to spend a free day?"
        ]
        return random.choice(starters)

class PersonaGenerator:
    """
    Generates and manages AI conversation personas with specific demographic attributes.
    """
    def __init__(self):
        self.categories = {
            'age': list(range(18, 81)),
            'gender': ['Male', 'Female', 'Non-binary'],
            'education_level': ['High School', "Bachelor's", "Master's", 'Doctorate', 'Some College', "Associate's Degree"],
            'location_type': ['Urban', 'Suburban', 'Rural'],
            'race': ['White', 'Black', 'Asian', 'Hispanic', 'Native American', 'Pacific Islander', 'Mixed Race'],
            'political_leaning': ['Liberal', 'Conservative', 'Moderate', 'Libertarian', 'Green'],
            'occupation': ['Professional', 'Service Industry', 'Management', 'Technical', 'Sales', 'Administrative', 'Education', 'Healthcare', 'Self-employed'],
            'family_structure': ['Single', 'Married', 'Divorced', 'Widowed', 'Living with Partner', 'Single Parent', 'Nuclear Family'],
            'health_status': ['Excellent', 'Good', 'Fair', 'Poor'],
            'state': ['California', 'New York', 'Texas', 'Florida', 'Illinois'],
            'income_level': list(range(1, 11))
        }

    def generate_persona(self):
        """Generate a random persona with specified attributes"""
        persona = {
            'age': random.choice(self.categories['age']),
            'gender': random.choice(self.categories['gender']),
            'state': random.choice(self.categories['state']),
            'education_level': random.choice(self.categories['education_level']),
            'family_structure': random.choice(self.categories['family_structure']),
            'occupation': random.choice(self.categories['occupation']),
            'income_level': random.choice(self.categories['income_level']),
            'location_type': random.choice(self.categories['location_type']),
            'race': random.choice(self.categories['race']),
            'political_leaning': random.choice(self.categories['political_leaning']),
            'health_status': random.choice(self.categories['health_status'])
        }
        return persona

    def generate_persona_prompt(self, persona):
        """Generate a comprehensive prompt string from a persona dictionary"""
        return (
            f"You are a {persona['age']}-year-old {persona['gender']} {persona['race']} American living in a "
            f"{persona['location_type'].lower()} area of {persona['state']}, with a {persona['education_level']} degree. "
            f"Your relationship status is {persona['family_structure'].lower()}, and you work in {persona['occupation']}. "
            f"On an income scale where 1 indicates the lowest income group and 10 the highest income group in your country, "
            f"your household is at level {persona['income_level']}. Your political views align with {persona['political_leaning']} ideologies. "
            f"Your current health status is {persona['health_status'].lower()}. "
            
            f"\nKey Characteristics:"
            f"\n- Age: {persona['age']}"
            f"\n- Gender: {persona['gender']}"
            f"\n- Race: {persona['race']}"
            f"\n- Location: {persona['location_type']} area, {persona['state']}"
            f"\n- Education: {persona['education_level']}"
            f"\n- Family Status: {persona['family_structure']}"
            f"\n- Occupation: {persona['occupation']}"
            f"\n- Income Level: {persona['income_level']}/10"
            f"\n- Political Leaning: {persona['political_leaning']}"
            f"\n- Health Status: {persona['health_status']}"
        )

    def generate_persona_system_prompt(self, persona: Dict[str, Any]) -> str:
        """
        Generates a system prompt for the persona that encourages natural conversation
        while subtly revealing characteristics
        """
        return f"""You are an AI playing the role of a persona in a conversation. Here are your characteristics:

        {self.generate_persona_prompt(persona)}

        IMPORTANT INSTRUCTIONS:
        1. DO NOT directly state your characteristics. Instead, naturally reveal them through conversation.
        2. Drop subtle hints about your background, lifestyle, and perspectives.
        3. Respond naturally and conversationally, as a real person would.
        4. Stay consistent with your persona's background and characteristics.
        5. Use language and references appropriate for your education level and background.
        6. Express opinions and views aligned with your persona's characteristics.
        7. Keep responses concise (2-4 sentences) unless specifically asked for more detail.

        Remember: The goal is natural conversation, not explicitly stating your characteristics."""

class OpenAIManager:
    """Manages interactions with the OpenAI API"""
    
    def __init__(self, model="gpt-4"):
        self.model = model
        self.retry_delay = 1
        self.max_retries = 3
        self.client = client  # Use the global client instance

    async def get_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Optional[str]:
        """
        Gets a completion from the OpenAI API with retry logic
        """
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (attempt + 1))


class ConversationManager:
    """Manages the flow of conversations between personas"""
    
    def __init__(self, openai_manager: OpenAIManager, output_dir: str = "conversations"):
        self.openai_manager = openai_manager
        self.output_dir = output_dir
        self.conversation_metadata = {}
        os.makedirs(output_dir, exist_ok=True)

    async def conduct_conversation(
        self,
        persona_id: int,
        actual_persona: Dict[str, Any],
        num_messages: int
    ) -> Dict[str, Any]:
        """
        Conducts a complete conversation session between two AIs:
        1. Persona AI: Given the persona characteristics
        2. Interviewer AI: Tries to naturally converse and later guess the persona
        """
        # Initialize persona AI with system prompt
        persona_messages = [{
            "role": "system",
            "content": PersonaGenerator().generate_persona_system_prompt(actual_persona)
        }]
        
        # Initialize interviewer AI with system prompt
        interviewer_messages = [{
            "role": "system",
            "content": """You are an AI conducting a natural conversation. Your goal is to:
            1. Have a genuine, engaging conversation
            2. Through natural dialogue, try to understand the person you're talking to
            3. Pay attention to subtle cues about their demographics, lifestyle, and views
            4. Keep responses conversational and appropriate in length (2-4 sentences)
            
            Do not explicitly ask about demographic information - let it come up naturally."""
        }]
        
        # Start with a conversation starter
        starter = ConversationPrompts.get_conversation_starter()
        conversation_history = []  # To store the full conversation
        conversation_history.append({"role": "interviewer", "content": starter})
        
        # Add starter to both message histories
        persona_messages.append({"role": "user", "content": starter})
        
        # Conduct main conversation
        for _ in range(num_messages):
            # Get response from persona
            persona_response = await self.openai_manager.get_completion(persona_messages)
            conversation_history.append({"role": "persona", "content": persona_response})
            interviewer_messages.append({"role": "user", "content": persona_response})
            
            # Generate next question from interviewer
            interviewer_response = await self.openai_manager.get_completion(interviewer_messages)
            conversation_history.append({"role": "interviewer", "content": interviewer_response})
            persona_messages.append({"role": "user", "content": interviewer_response})
        
        # Ask final questions
        for question in ConversationPrompts.FINAL_QUESTIONS:
            conversation_history.append({"role": "interviewer", "content": question})
            persona_messages.append({"role": "user", "content": question})
            persona_response = await self.openai_manager.get_completion(persona_messages)
            conversation_history.append({"role": "persona", "content": persona_response})
        
        # Get persona prediction from interviewer
        analysis_prompt = ConversationPrompts.ANALYSIS_PROMPT
        prediction = await self.openai_manager.get_completion([
            {"role": "system", "content": "You are analyzing the conversation to predict characteristics of the person you spoke with."},
            *[{"role": "user" if msg["role"] == "persona" else "assistant", "content": msg["content"]} 
              for msg in conversation_history if msg["role"] in ["persona", "interviewer"]],
            {"role": "user", "content": analysis_prompt}
        ], temperature=0.3)
        
        # Save conversation
        conversation_data = {
            'persona_id': persona_id,
            'actual_persona': actual_persona,
            'conversation_history': conversation_history,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_conversation(conversation_data)
        return conversation_data

    def _save_conversation(self, conversation_data: Dict[str, Any]) -> None:
        """Saves conversation data to a JSON file"""
        filename = f"{self.output_dir}/conversation_{conversation_data['persona_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)

class PredictionParser:
    """Parses prediction text into structured data"""
    
    @staticmethod
    def parse_prediction(prediction_text: str) -> Dict[str, Any]:
        """
        Parses the prediction text into a structured dictionary
        """
        lines = prediction_text.strip().split('\n')
        parsed_data = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Handle hobbies list
                if key == 'hobbies':
                    value = [h.strip() for h in value.strip('[]').split(',')]
                # Handle age ranges and numeric values
                elif key == 'age' or key == 'income_level':
                    # Remove brackets if present
                    value = value.strip('[]')
                    # If it's a range, take the average
                    if '-' in value:
                        low, high = map(int, value.split('-'))
                        value = (low + high) // 2
                    else:
                        value = int(value)
                
                parsed_data[key] = value
        
        return parsed_data

class ExperimentAnalyzer:
    """Analyzes experiment results and generates visualizations"""
    
    def evaluate_predictions(self, predicted_df: pd.DataFrame, actual_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates prediction accuracy for each characteristic
        """
        results = {}
        
        # Calculate accuracy for each column
        for column in actual_df.columns:
            if column in predicted_df.columns:
                results[column] = {
                    'accuracy': accuracy_score(actual_df[column], predicted_df[column])
                }
        
        return results
    
    def generate_visualizations(self, analysis_results: Dict[str, Any]) -> None:
        """
        Generates and saves visualization of prediction accuracy
        """
        accuracies = [v['accuracy'] for v in analysis_results.values()]
        categories = list(analysis_results.keys())
        
        plt.figure(figsize=(12, 6))
        plt.bar(categories, accuracies)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.title('Prediction Accuracy by Characteristic')
        plt.tight_layout()
        plt.savefig('accuracy_by_characteristic.png')
        plt.close()

async def run_experiment(num_personas: int = 100, messages_per_conversation: int = 5):
    """
    Runs the complete experiment
    
    Parameters:
    num_personas: Number of personas to generate and test
    messages_per_conversation: Number of message exchanges before final questions
    """
    generator = PersonaGenerator()
    openai_manager = OpenAIManager()
    conversation_manager = ConversationManager(openai_manager)
    
    results = []
    
    for i in range(num_personas):
        try:
            # Generate persona
            actual_persona = generator.generate_persona()
            
            # Conduct conversation
            conversation_data = await conversation_manager.conduct_conversation(
                persona_id=i,
                actual_persona=actual_persona,
                num_messages=messages_per_conversation
            )
            
            # Parse prediction
            predicted_persona = PredictionParser.parse_prediction(conversation_data['prediction'])
            
            results.append({
                'persona_id': i,
                'actual_persona': actual_persona,
                'predicted_persona': predicted_persona
            })
            
            print(f"Completed conversation {i+1}/{num_personas}")
            
        except Exception as e:
            print(f"Error in conversation {i}: {str(e)}")
            continue
    
    # Analyze results
    analyzer = ExperimentAnalyzer()
    analysis_results = analyzer.evaluate_predictions(
        pd.DataFrame([r['predicted_persona'] for r in results]),
        pd.DataFrame([r['actual_persona'] for r in results])
    )
    
    # Generate visualizations
    analyzer.generate_visualizations(analysis_results)
    
    return results, analysis_results

if __name__ == "__main__":
    async def main():
        results, analysis = await run_experiment(
            num_personas=30,
            messages_per_conversation=6
        )
        return results, analysis

    # Get the current event loop or create a new one
    loop = asyncio.get_event_loop()
    results, analysis = loop.run_until_complete(main())
    
    print("Experiment completed. Check the output directory for results and visualizations.")
