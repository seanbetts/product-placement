import os
import json
import time
import requests
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load .env file
load_dotenv(os.path.join(script_dir, '.env'))

SAM2_URL = os.getenv("SAM2_URL")
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")

# Check if all required environment variables are set
required_vars = ["SAM2_URL", "PROJECT_ID", "BUCKET_NAME", "SERVICE_ACCOUNT_FILE"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print("Error: The following required environment variables are not set:")
    for var in missing_vars:
        print(f"- {var}")
    print("Please set these variables in your .env file or environment.")
    exit(1)

# If SERVICE_ACCOUNT_FILE is a relative path, make it absolute
if not os.path.isabs(SERVICE_ACCOUNT_FILE):
    SERVICE_ACCOUNT_FILE = os.path.join(script_dir, SERVICE_ACCOUNT_FILE)

if not os.path.exists(SERVICE_ACCOUNT_FILE):
    print(f"Error: Service account key file not found at {SERVICE_ACCOUNT_FILE}")
    print("Please ensure the path in GOOGLE_APPLICATION_CREDENTIALS is correct.")
    exit(1)

def test_segment_video_gcs(video_id):
    url = f"{SAM2_URL}/segment_video_gcs/{video_id}"
    
    response = requests.post(url)
    
    if response.status_code == 200:
        print("Segment Video GCS Response:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def wait_for_results(video_id, max_attempts=10, delay=30):
    for attempt in range(max_attempts):
        print(f"Checking for results (attempt {attempt + 1}/{max_attempts})...")
        if check_gcs_results(video_id):
            return True
        time.sleep(delay)
    print("Max attempts reached. Results not found.")
    return False

def check_gcs_results(video_id):
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        client = storage.Client(credentials=credentials, project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        
        result_blob_name = f"{video_id}/sam2_result.json"
        blob = bucket.blob(result_blob_name)
        
        if blob.exists():
            print("Results found in GCS:")
            results = json.loads(blob.download_as_string())
            print(json.dumps(results[:5], indent=2))  # Print first 5 results
            return True
        else:
            print("Results not found in GCS. Processing may still be ongoing.")
            return False
    except Exception as e:
        print(f"Error checking GCS results: {str(e)}")
        return False

if __name__ == "__main__":
    video_id = input("Enter the video ID to process: ")
    test_segment_video_gcs(video_id)
    
    wait_for_results(video_id)