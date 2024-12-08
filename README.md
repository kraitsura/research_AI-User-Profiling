# AI User Profiling Research

## Overview
This research project investigates whether AI systems can accurately predict user personas through simulated conversations. The experiment involves generating conversations between AI agents and analyzing their ability to infer demographic and psychographic characteristics of simulated personas.

## Project Structure
- `main.py`: Core experiment runner and analysis
- `conversations/`: Directory containing JSON files of simulated conversations
  - Each conversation includes:
    - Actual persona attributes
    - Conversation history
    - AI's prediction
    - Timestamp

## Key Components

### Persona Attributes
The experiment tracks multiple demographic and psychographic variables:
- Age
- Gender
- Race
- Education Level
- Family Structure
- Occupation
- Income Level
- Location Type
- State
- Political Leaning
- Health Status

### Methodology
1. Generate diverse personas with specific attributes
2. Simulate conversations between an interviewer and the persona
3. Analyze AI's ability to predict the persona's attributes
4. Evaluate accuracy and patterns in predictions

### Technical Stack
- Python
- OpenAI API
- Pandas for data analysis
- Scikit-learn for metrics
- Matplotlib/Seaborn for visualizations

## Running the Experiment
The experiment is run by calling the `main()` function in `main.py`.
```python
async def main():
    results, analysis = await run_experiment(
        num_personas=30,
        messages_per_conversation=6
    )  
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-user-profiling.git
cd ai-user-profiling
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'  # On Windows, use: set OPENAI_API_KEY=your-api-key-here
```

### Running
To run the experiment:
```bash
python main.py
```

## Data Collection
Conversations are stored as JSON files with:
- Persona attributes
- Conversation history
- AI predictions
- Timestamps

## Analysis
The project analyzes:
- Prediction accuracy across different attributes
- Patterns in misclassification
- Demographic bias in predictions
- Conversation quality metrics

## Future Work
- Expand the range of persona attributes
- Analyze conversation patterns
- Study bias in AI predictions
- Improve prediction accuracy

## License
MIT
