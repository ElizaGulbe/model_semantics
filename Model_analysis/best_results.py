import os
import json
import pandas as pd
import pprint  # Import pprint for pretty-printing
from sklearn.linear_model import LinearRegression

# Set the path to the directory containing the runs
results_dir = "Ray_results/First run"
# Initialize an empty list to store results
data = []

# Loop through each subdirectory (each run) and collect results
for run_dir in os.listdir(results_dir):
    run_path = os.path.join(results_dir, run_dir)
    results_file = os.path.join(run_path, 'result.json')  # Updated to correct filename
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract metrics
        metrics = {
            'accuracy': results.get('accuracy'),
            'precision': results.get('precision'),
            'recall': results.get('recall'),
            'f1_score': results.get('f1_score'),
            'confusion_matrix': results.get('confusion_matrix'),
            'classification_report': results.get('classification_report')
        }
        
        # Extract configuration
        config = results.get('config', {})
        
        # Convert hidden_sizes list to string for easier handling
        if 'hidden_sizes' in config:
            config['hidden_sizes'] = str(config['hidden_sizes'])
        
        # Combine metrics and config
        combined = {**metrics, 'config': config}
        data.append(combined)

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
if df.empty:
    print("DataFrame is empty. No data was collected.")
else:
    # Save the DataFrame to a CSV file for further analysis if needed
    df.to_csv('model_results_analysis.csv', index=False)

    # Find the result with the highest accuracy
    best_result = df.loc[df['accuracy'].idxmax()]
    
    print("\nBest Result:")

    # Pretty-print the best result
    pprint.pprint(best_result.to_dict(), width=100)

    # Alternatively, manually format the output for better readability
    print("\nBest Result (Formatted):")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall: {best_result['recall']:.4f}")
    print(f"F1 Score: {best_result['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    for row in best_result['confusion_matrix']:
        print(f"{row}")
    
    print("\nClassification Report:")
    print(best_result['classification_report'])
    
    print("\nConfiguration:")
    for key, value in best_result['config'].items():
        print(f"  {key}: {value}")

    # Descriptive statistics (optional: if you need to calculate them later)
    # ...
