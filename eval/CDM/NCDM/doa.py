import pandas as pd
import numpy as np
import pickle
import json

def calculate_doa_k(mas_level, q_matrix, r_matrix, k):
    n_students, n_skills = mas_level.shape
    n_questions, _ = q_matrix.shape
    DOA_k = 0.0
    numerator = 0
    denominator = 0

    delta_matrix = mas_level[:, k].reshape(-1, 1) > mas_level[:, k].reshape(1, -1)
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()

    for j in question_hask:
        row_vector = (r_matrix[:, j].reshape(1, -1) != -1).astype(int)
        column_vector = (r_matrix[:, j].reshape(-1, 1) != -1).astype(int)
        mask = row_vector * column_vector
        delta_r_matrix = r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)
        I_matrix = r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)

        numerator_ = np.logical_and(mask, delta_r_matrix)
        denominator_ = np.logical_and(mask, I_matrix)

        numerator += np.sum(delta_matrix * numerator_)
        denominator += np.sum(delta_matrix * denominator_)

    if denominator != 0:
        DOA_k = numerator / denominator
        return DOA_k
    else:
        return 0

def process_response_matrix(json_file, n_students, n_questions):
    # Initialize response matrix with -1 (indicating no attempt)
    r_matrix = -1 * np.ones((n_students, n_questions))

    # Load response data from JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Fill response matrix
    for entry in data:
        user_id = entry["user_id"] - 1  # Assuming user_id starts from 1
        logs = entry["logs"]
        for log in logs:
            exer_id = log["exer_id"] - 1  # Assuming exer_id starts from 1
            score = log["score"]
            r_matrix[user_id, exer_id] = score  # Fill the score

    return r_matrix

def compute_average_doa(mas_level_file, q_matrix_file, response_file):
    # Load mas_level from the pickle file
    with open(mas_level_file, 'rb') as f:
        mas_level = pickle.load(f)
    print(mas_level.shape)
    # Load q_matrix from CSV file
    q_matrix = pd.read_csv(q_matrix_file, header=None).values
    print(q_matrix.shape)
    # Get dimensions
    n_students, n_skills = mas_level.shape
    n_questions, _ = q_matrix.shape

    # Process response matrix from JSON file
    r_matrix = process_response_matrix(response_file, n_students, n_questions)
    print(r_matrix.shape)
    # Compute DOA for each skill and calculate the average
    doa_values = []
    for k in range(n_skills):
        doa_k = calculate_doa_k(mas_level, q_matrix, r_matrix, k)
        doa_values.append(doa_k)
    print(doa_values)
    # Compute average DOA
    average_doa = np.mean(doa_values)
    return average_doa

if __name__ == "__main__":
    # File paths
    mas_level_file = 'result/student_stat.pkl' # Replace with the actual path to the pkl file
    q_matrix_file = '../ncd_data/qmatrix.csv'  # Replace with the actual path to the qmatrix CSV file
    response_file = '../ncd_data/ncd_data.json'  # Replace with the actual path to the response JSON file

    # Compute the average DOA
    avg_doa = compute_average_doa(mas_level_file, q_matrix_file, response_file)
    print(f"Average DOA: {avg_doa}")