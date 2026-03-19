import os
import json
from openai import OpenAI
from src.loadenv import load_env_vars

load_env_vars()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=f"{openai_api_key}")

def extract_claims_from_dir(directory_path, output_file="all_extracted_claims.json"):
    all_results = {}

    # 1. Loop through every file in the folder
    for filename in os.listdir(directory_path):
        # We only want to process text or markdown files for now
        if filename.endswith((".txt", ".md")):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing: {filename}...")

            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()

            try:
                # 2. Make the API call for this specific file
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system", 
                            "content": "Extract every environmental or sustainability claim. Output as a JSON object with a key 'claims' containing a list of strings."
                        },
                        {"role": "user", "content": document_text}
                    ],
                    response_format={ "type": "json_object" }
                )
                
                # 3. Store the result using the filename as the key
                data = json.loads(response.choices[0].message.content)
                all_results[filename] = data.get("claims", [])

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # 4. Save everything into one master JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n Done! All claims saved to {output_file}")
    return all_results

# Usage
my_dir = os.gevenv("CLAIMS_SOURCE")
extract_claims_from_dir(my_dir)

