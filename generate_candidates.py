import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle

def serialize_tensor(tensor):
    return pickle.dumps(tensor).hex()

def deserialize_tensor(serialized_tensor):
    return pickle.loads(bytes.fromhex(serialized_tensor))

def compute_top_euclidean_similarities_for_row(tf_tensor, row_index, top_n=100):
    # Compute the Euclidean distance between a single row and all rows
    target_tensor = tf_tensor[row_index:row_index+1]
    squared_tensor = tf.reduce_sum(tf.square(tf_tensor), axis=1, keepdims=True)
    target_squared = tf.reduce_sum(tf.square(target_tensor), axis=1, keepdims=True)
    
    euclidean_distance_vector = squared_tensor + target_squared - 2 * tf.matmul(tf_tensor, tf.transpose(target_tensor))
    euclidean_distance_vector = tf.sqrt(tf.maximum(euclidean_distance_vector, 0.0))  # Ensure no negative values

    # Set the self-distance to +inf to exclude it from being selected
    update_shape = (1, 1)  # Ensure the update is correctly shaped as (1, 1)
    euclidean_distance_vector = tf.tensor_scatter_nd_update(
        euclidean_distance_vector,
        indices=[[row_index]],
        updates=tf.constant([np.inf], dtype=euclidean_distance_vector.dtype, shape=update_shape)
    )

    # Get the top N indices with the smallest distances (i.e., most similar)
    top_indices = tf.argsort(euclidean_distance_vector[:, 0], direction='ASCENDING')[:top_n]

    return top_indices.numpy()
relations_similarity = []

def add_similarity_relation(row1, row2): 
    new_row = {
        'sense1_id': row1["sense_id"],
        'sense1_entry_id': row1["entry_id"],
        'sense1_heading': row1['entry_heading'],
        'sense1_gloss': row1['gloss'],
        'sense2_id': row2["sense_id"],
        'sense2_entry_id': row2["entry_id"],
        'sense2_heading': row2['entry_heading'],
        'sense2_gloss': row2['gloss'],
        'sense1_gloss_embedding': serialize_tensor(row1["gloss_embedding"]),
        'sense2_gloss_embedding': serialize_tensor(row2["gloss_embedding"]),
    }
    relations_similarity.append(new_row)

# Load and prepare the DataFrame
df = pd.read_csv("Production/Generate candidates/nouns_with_embeddings.csv")
df["gloss_embedding"] = df["gloss_embedding"].apply(deserialize_tensor)
df = df.drop_duplicates(subset=['sense_id'], keep='first')
df = df.reset_index(drop=True)

# Convert gloss embeddings to TensorFlow tensors
tensor_list = df['gloss_embedding'].tolist()
tensor_list_reduce_dim = [item[0] for item in tensor_list] 
tf_tensor = tf.stack(tensor_list_reduce_dim)

# Loop through the first 200 rows and compute top similarities for each row
for i in tqdm(range(0,200)):
    top_similarities_indices = compute_top_euclidean_similarities_for_row(tf_tensor, i, top_n=100)
    # Add the similarity relations to the list
    [add_similarity_relation(df.loc[i, :], df.loc[int(idx), :]) for idx in top_similarities_indices]

# Convert the list of relations to a DataFrame and save it
relations_df = pd.DataFrame(relations_similarity)
relations_df.to_csv("Production/Generate candidates/candidate_files.csv", index=False)
