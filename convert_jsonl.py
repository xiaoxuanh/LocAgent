import json

def convert_jsonl(input_file, output_file):
    """
    Convert source JSONL format to target JSONL format.
    
    The main differences are:
    1. Source has complex nested arrays in found_files, found_modules, found_entities
    2. Target has simple flat arrays
    3. Need to flatten nested structures while preserving the data
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.strip():  # Skip empty lines
                data = json.loads(line.strip())
                
                # Convert the data structure
                converted_data = {
                    "instance_id": data["instance_id"],
                    "found_files": flatten_nested_array(data.get("found_files", [])),
                    "found_modules": flatten_nested_array(data.get("found_modules", [])),
                    "found_entities": flatten_nested_array(data.get("found_entities", [])),
                    "raw_output_loc": data.get("raw_output_loc", []),
                    "meta_data": data.get("meta_data", {})
                }
                
                outfile.write(json.dumps(converted_data) + '\n')

def flatten_nested_array(nested_array):
    """
    Flatten a nested array structure to a simple array.
    If the input is already flat, return as is.
    If it contains nested arrays, flatten them.
    """
    if not nested_array:
        return []
    
    # Check if this is already a flat array of strings
    if all(isinstance(item, str) for item in nested_array):
        return nested_array
    
    # If it contains nested arrays, flatten them
    flattened = []
    for item in nested_array:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    
    return flattened

if __name__ == "__main__":
    input_file = "/home/azureuser/cloudfiles/code/Users/v-xhou/locagent/LocAgent/results/location/loc_outputs.jsonl"
    output_file = "/home/azureuser/cloudfiles/code/Users/v-xhou/locagent/LocAgent/results/location/converted_loc_outputs.jsonl"

    convert_jsonl(input_file, output_file)
    print(f"Conversion complete. Output written to: {output_file}")
