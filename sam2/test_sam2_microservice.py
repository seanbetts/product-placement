import requests
import os
import json
from google.cloud import storage
from google.oauth2 import service_account

# SAM 2 microservice URL
SAM2_URL = "https://sam2-583054821893.europe-west2.run.app"

# Google Cloud Storage settings
PROJECT_ID = "product-placement-analytics"
BUCKET_NAME = "your-test-bucket-name"
GCS_VIDEO_PATH = "path/to/your/test/video.mp4"

# Path to your service account key file
SERVICE_ACCOUNT_FILE = "path/to/your/service-account-key.json"

def test_segment_video():
    """Test the /segment_video endpoint with a local video file."""
    url = f"{SAM2_URL}/segment_video"
    video_path = "path/to/your/local/test/video.mp4"
    
    with open(video_path, "rb") as video_file:
        files = {"file": ("test_video.mp4", video_file, "video/mp4")}
        response = requests.post(url, files=files)
    
    print("Segment Video Response:")
    print(response.json())
    
    # You may want to add a loop here to periodically check for the results
    # as the processing happens asynchronously

def test_segment_video_gcs():
    """Test the /segment_video_gcs endpoint with a video in Google Cloud Storage."""
    url = f"{SAM2_URL}/segment_video_gcs"
    
    params = {
        "bucket_name": BUCKET_NAME,
        "blob_name": GCS_VIDEO_PATH
    }
    
    response = requests.post(url, params=params)
    
    print("Segment Video GCS Response:")
    print(response.json())
    
    # You may want to add a loop here to periodically check for the results
    # as the processing happens asynchronously

def check_gcs_results():
    """Check Google Cloud Storage for the results of video processing."""
    # Authenticate with Google Cloud
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # Create a Google Cloud Storage client
    client = storage.Client(credentials=credentials, project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    # Construct the expected result blob name
    result_blob_name = f"{os.path.splitext(GCS_VIDEO_PATH)[0]}_sam2_result.json"
    blob = bucket.blob(result_blob_name)
    
    if blob.exists():
        print("Results found in GCS:")
        results = json.loads(blob.download_as_string())
        print(json.dumps(results, indent=2))
    else:
        print("Results not found in GCS. Processing may still be ongoing.")

if __name__ == "__main__":
    test_segment_video()
    test_segment_video_gcs()
    
    # Wait for a bit before checking results (adjust as needed)
    import time
    print("Waiting for 60 seconds before checking results...")
    time.sleep(60)
    
    check_gcs_results()