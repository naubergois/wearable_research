
import json
import os

def main():
    input_file = "figures_metadata.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r") as f:
        data = json.load(f)

    # Split into separate files
    files_mapping = {
        "figure_10": "metadata_figure_10.json",
        "time_domain_signals": "metadata_time_domain.json",
        "figure_11": "metadata_figure_11.json"
    }

    for key, filename in files_mapping.items():
        if key in data:
            print(f"Exporting {key} to {filename}...")
            with open(filename, "w") as out:
                json.dump(data[key], out, indent=4)
        else:
            print(f"Warning: Key {key} not found in metadata.")

if __name__ == "__main__":
    main()
