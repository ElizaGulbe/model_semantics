import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
            'f1_score': results.get('f1_score')
        }
        
        # Extract configuration
        config = results.get('config', {})
        
        # Convert hidden_sizes list to string for easier handling
        if 'hidden_sizes' in config:
            config['hidden_sizes'] = str(config['hidden_sizes'])
        
        # Combine metrics and config
        combined = {**metrics, **config}
        data.append(combined)

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
if df.empty:
    print("DataFrame is empty. No data was collected.")
else:
    # Save the DataFrame to a CSV file for further analysis if needed
    df.to_csv('model_results_analysis.csv', index=False)

    # Descriptive statistics
    print("Descriptive Statistics:")
    print(df.describe())

    # Separate numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Correlation analysis on numeric columns only
    correlation_matrix = numeric_df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Visualize the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix between Parameters and Metrics')
    plt.show()

    # Scatter plots for continuous parameters
    continuous_params = ['lr', 'num_epochs', 'batch_size', 'dropout_rate', 'postive_percentage']
    for param in continuous_params:
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df[param], y=df[metric])
            plt.title(f'{metric} vs {param}')
            plt.xlabel(param)
            plt.ylabel(metric)
            plt.show()

    # Box plots for categorical parameters including hidden_sizes
    categorical_params = ['activation', 'optimizer', 'hidden_sizes']
    for param in categorical_params:
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[param], y=df[metric])
            plt.title(f'{metric} by {param}')
            plt.xlabel(param)
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.show()

    # Linear regression analysis on numeric columns
    for param in continuous_params:
        X = df[[param]].values
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            y = df[metric].values
            model = LinearRegression().fit(X, y)
            score = model.score(X, y)
            print(f'Regression R^2 for {metric} vs {param}: {score:.2f}')
            
            # Plot the regression line
            plt.figure(figsize=(8, 6))
            sns.regplot(x=df[param], y=df[metric], ci=None, line_kws={"color": "red"})
            plt.title(f'{metric} vs {param} with Regression Line')
            plt.xlabel(param)
            plt.ylabel(metric)
            plt.show()

    print("Analysis complete.")
