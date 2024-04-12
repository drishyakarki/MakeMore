import pandas as pd
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Extract data from a CSV file and write it to a text file")
    parser.add_argument("--data", type=str, help="Input dataset in CSV format")
    args = parser.parse_args()

    if not args.data:
        parser.error("Please provide the input dataset using --data")

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        print(f"Error: File '{args.data}' not found.")
        exit(1)
    
    names = df['name'].tolist()

    file_name = args.data.split('.')[0]

    with open(f"{file_name}.txt", 'w') as file:
        for name in names:
            file.write(f"{name}\n")
