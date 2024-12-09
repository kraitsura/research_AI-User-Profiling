import json
import os
import pandas as pd
from typing import List, Dict, Any
from main import ExperimentAnalyzer as MainAnalyzer, PredictionParser as MainParser
from visual import ExperimentAnalyzer as VisualAnalyzer, PredictionParser as VisualParser

def load_conversations(conversations_dir: str) -> List[Dict[str, Any]]:
    """Load all conversation JSON files from the specified directory"""
    results = []
    for filename in os.listdir(conversations_dir):
        if filename.endswith('.json'):
            with open(os.path.join(conversations_dir, filename), 'r') as f:
                try:
                    conversation_data = json.load(f)
                    # Parse the prediction text into structured data
                    predicted_persona = MainParser.parse_prediction(conversation_data['prediction'])
                    results.append({
                        'persona_id': conversation_data['persona_id'],
                        'actual_persona': conversation_data['actual_persona'],
                        'predicted_persona': predicted_persona
                    })
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    return results

def main():
    # Create output directories for both analyses
    os.makedirs('analysis_output', exist_ok=True)
    os.makedirs('conversation_analysis', exist_ok=True)

    # Load conversations
    print("Loading conversations...")
    conversations_dir = 'conversations'
    results = load_conversations(conversations_dir)

    if not results:
        print("No conversations found to analyze!")
        return

    print(f"Loaded {len(results)} conversations for analysis")

    # Create DataFrames for analysis
    actual_df = pd.DataFrame([r['actual_persona'] for r in results])
    predicted_df = pd.DataFrame([r['predicted_persona'] for r in results])

    # Run main analyzer
    print("\nRunning main analyzer...")
    main_analyzer = MainAnalyzer()
    main_results = main_analyzer.evaluate_predictions(predicted_df, actual_df)
    main_analyzer.generate_visualizations(main_results)
    print("Main analysis complete! Check the analysis_output directory for results.")

    # Run visual analyzer
    print("\nRunning visual analyzer...")
    visual_analyzer = VisualAnalyzer(output_dir="conversation_analysis")
    visual_results = visual_analyzer.evaluate_predictions(predicted_df, actual_df)
    visual_analyzer.generate_visualizations(actual_df, predicted_df, visual_results)
    
    # Print results from visual analyzer
    print("\nVisual Analysis Results:")
    for characteristic, results in visual_results.items():
        print(f"{characteristic}: {results['accuracy']:.2%} accuracy")
    
    print("\nBoth analyses complete!")
    print("- Main analysis results in 'analysis_output' directory")
    print("- Visual analysis results in 'conversation_analysis' directory")

if __name__ == "__main__":
    main() 