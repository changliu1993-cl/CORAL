import os
import json


def format_file(dir):
    '''
    Format the file to make it a single JSON array
    '''
    files = os.listdir(dir)
    files = [file for file in files if '.jsonl' in file]

    for file in files:
        with open(os.path.join(dir, file), 'r') as f:
            file_content = f.read()

        # Replace '][' with ',' to make it a single JSON array
        if '][' in file_content:
            modified_content = file_content.replace('][', ',')

            # Parse the modified string as JSON
            try:
                data = json.loads(modified_content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                data = []

            # write to the same file
            try:
                assert len(data) == 500 or len(data) == 3000
            except:
                print(f"Error: {file} has {len(data)} lines")
            # save_file = file.replace('.jsonl', '.json')
            with open(os.path.join(dir, file), 'w') as f:
                json.dump(data, f, indent=4)
            print(f"save to {os.path.join(dir, file)}")         


def format_dict_file(dir):
    '''
    Format the file to make it a single JSON array
    '''
    files = os.listdir(dir)
    files = [file for file in files if '.json' in file]

    for file in files:
        with open(os.path.join(dir, file), 'r') as f:
            file_content = f.read()

        # Replace '][' with ',' to make it a single JSON array
        if '}{' in file_content:
            modified_content = file_content.replace('}{', '},{')
            modified_content = '[' + modified_content + ']'
            # Parse the modified string as JSON
            try:
                data = json.loads(modified_content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                data = []

            # write to the same file
            with open(os.path.join(dir, file), 'w') as f:
                json.dump(data, f, indent=4)
            print(f"save to {os.path.join(dir, file)}")         