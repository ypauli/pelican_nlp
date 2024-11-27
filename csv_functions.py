import csv
def store_features_to_csv(per_token_data, results_csv_path):
    # Write the data to the CSV file
    headers = per_token_data[0].keys()
    with open(results_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(per_token_data)
    print(f"Data successfully saved to {results_csv_path}")
    return