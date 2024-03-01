import os

def get_empty_files(folder_path):
    empty_files = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return empty_files

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the path is a file and if the file is empty
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            empty_files.append(filename)

    return empty_files

# Example usage
folder_path = 'zod/labels/'
empty_files = get_empty_files(folder_path)

if empty_files:
    print("Empty files found:")
    for file_name in empty_files:
        print(file_name)
    print(len(empty_files), "empty files found in the folder", folder_path)
else:
    print(f"There are no empty files in the folder '{folder_path}'.")

