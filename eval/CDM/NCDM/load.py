import torch
import numpy as np
import pickle
import csv
import sys
from model import Net

def load_snapshot(model, filename):
    """Load the trained model snapshot."""
    with open(filename, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))

def get_status(student_n, exer_n, knowledge_n, epoch=5, output_pkl='result/student_stat.pkl'):
    """
    Get and save students' knowledge statuses (theta values) as a NumPy array using pickle.

    Parameters:
    - student_n: Number of students
    - exer_n: Number of exercises
    - knowledge_n: Number of knowledge concepts
    - epoch: The epoch number of the trained model to load (default is 5)
    - output_pkl: Path to save the resulting theta values as a pickle file
    """
    model_path = f'model/model_epoch{epoch}'  # Construct model path based on epoch
    net = Net(student_n, exer_n, knowledge_n)
    
    try:
        load_snapshot(net, model_path)  # Load the model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
    
    net.eval()
    
    # List to store the theta values for all students
    theta_list = []
    
    # Retrieve the knowledge status for each student
    for stu_id in range(student_n):
        # Get knowledge status of student with stu_id (index)
        status = net.get_knowledge_status(torch.LongTensor([stu_id])).tolist()[0]
        theta_list.append(status)
    
    # Convert the list of statuses into a NumPy array
    theta_mat = np.array(theta_list)
    
    # Save the NumPy array to a pickle file
    with open(output_pkl, 'wb') as f:
        pickle.dump(theta_mat, f)
    
    print(f"Theta values saved to {output_pkl}")

def load_pkl_to_txt_and_csv(pkl_file, txt_file, csv_file):
    """
    Loads data from a pickle (.pkl) file and writes the numerical values (truncated to 5 digits)
    to both a text (.txt) file and a CSV (.csv) file. Each line in the array will be written as a 
    line in both files.

    Args:
        pkl_file (str): Path to the input pickle file.
        txt_file (str): Path to the output text file.
        csv_file (str): Path to the output CSV file.
    """
    # Load the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Ensure the data is iterable (list, ndarray, etc.)
    if isinstance(data, (list, tuple)) or hasattr(data, "__iter__"):
        # Helper function to truncate numbers to 5 digits
        def truncate_to_5_digits(value):
            try:
                # Check if the value is numerical
                if isinstance(value, (int, float)):
                    return round(value, 5)  # Keep up to 5 digits after the decimal
                return value  # For non-numerical values, return as is
            except Exception as e:
                return value  # In case of unexpected issues, return the value as is
        
        # Write to the text file
        with open(txt_file, 'w') as txt_f:
            for row in data:
                # Truncate each number in the row to 5 digits
                truncated_row = [truncate_to_5_digits(x) for x in row]
                # Convert each row to a space-separated string and write to the text file
                txt_f.write(" ".join(map(str, truncated_row)) + "\n")
        
        # Write to the CSV file
        with open(csv_file, 'w', newline='') as csv_f:
            writer = csv.writer(csv_f)
            # Truncate each number in each row to 5 digits
            truncated_data = [[truncate_to_5_digits(x) for x in row] for row in data]
            writer.writerows(truncated_data)  # Write rows directly to the CSV file
        
        print(f"Data from {pkl_file} has been written to {txt_file} and {csv_file}")
    else:
        print("Unsupported data format. Make sure the pickle file contains iterable data.")

if __name__ == '__main__':
    # Load configuration from 'config.txt'
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = map(eval, i_f.readline().split(','))

    # Get 'epoch' argument from command line (default to 5 if not provided)
    epoch = 5  # Default model epoch
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        epoch = int(sys.argv[1])

    # Define file paths
    output_pkl = 'result/student_stat.pkl'
    output_txt = 'result/output.txt'
    output_csv = 'result/result.csv'

    # Run the get_status function to generate the pickle file
    get_status(student_n, exer_n, knowledge_n, epoch, output_pkl)

    # Convert the pickle file to text and CSV formats
    load_pkl_to_txt_and_csv(output_pkl, output_txt, output_csv)