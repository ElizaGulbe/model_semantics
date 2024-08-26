import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set the path to the directory containing the runs
results_dir = r'C:\ray_results\train_model_2024-08-24_20-00-55'

# Initialize an empty list to store results
data = []

# Loop through each subdirectory (each run) and collect results
for run_dir in os.listdir(results_dir):
    run_path = os.path.join(results_dir, run_dir)
    results_file = os.path.join(run_path, 'result.json')
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            try:
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
                
                # Convert hidden_sizes to a string for categorization
                hidden_sizes = str(config.get('hidden_sizes', []))
                
                # Combine metrics and config
                combined = {**metrics, **config, 'hidden_sizes': hidden_sizes}
                data.append(combined)
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file: {results_file} - {e}")

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
if df.empty:
    print("DataFrame is empty. No data was collected.")
else:
    print(f"Data collected for {len(df)} runs.")
    print(df.head())

# Save the DataFrame to a CSV file for further analysis if needed
df.to_csv('model_results_analysis.csv', index=False)

# Descriptive statistics
print("Descriptive Statistics:")
print(df.describe())

# Correlation analysis
correlation_matrix = df.corr()
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

# Box plots for categorical parameters
categorical_params = ['activation', 'optimizer']
for param in categorical_params:
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[param], y=df[metric])
        plt.title(f'{metric} by {param}')
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.show()

# Linear regression analysis
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

# Distribution of accuracy for different hidden_sizes configurations
plt.figure(figsize=(14, 8))
sns.boxplot(x='hidden_sizes', y='accuracy', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Accuracy for Different Hidden Sizes Configurations')
plt.xlabel('Hidden Sizes')
plt.ylabel('Accuracy')
plt.show()

print("Analysis complete.")
