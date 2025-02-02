import json
import time
import os
from openai import OpenAI  # Import the OpenAI client
from app import fetch_all_data  # Import your custom function

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def prepare_finetune_data():
    # Fetch data from the database
    conv_data, feedback_data = fetch_all_data()

    # Prepare training examples in chat format
    training_examples = []
    for conv in conv_data:
        example = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": conv["question"]},
                {"role": "assistant", "content": conv["response"]}
            ]
        }
        training_examples.append(example)

    # Save training data to a JSONL file
    with open("training_data.jsonl", "w", encoding="utf-8") as f:
        for ex in training_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# Call the function to prepare data
prepare_finetune_data()

# Upload the training file to OpenAI
try:
    upload_response = client.files.create(
        file=open("training_data.jsonl", "rb"),
        purpose="fine-tune"
    )
    training_file_id = upload_response.id
    print(f"The file has been successfully uploaded. File ID: {training_file_id}")
except Exception as e:
    print(f"An error occurred while uploading the file: {e}")
    exit()

# Wait for the file to be processed
file_ready = False
while not file_ready:
    try:
        file_status = client.files.retrieve(training_file_id)
        if file_status.status == "processed":
            file_ready = True
            print("File processing complete.")
        else:
            print("Processing the file...")
            time.sleep(5)
    except Exception as e:
        print(f"An error occurred while checking the file status: {e}")
        exit()

# Start the fine-tuning job
try:
    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-3.5-turbo-0125"  # Use the latest supported model
    )
    print("The fine-tuning task has been initiated. Job ID:", fine_tune_response.id)
except Exception as e:
    print(f"An error occurred while starting the fine-tuning task: {e}")