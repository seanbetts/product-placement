backend/
│
├── .env
│
├── .env.example
│
├── Dockerfile
│
├── README.md
│
├── __pycache__
│
├── api
│   │
│   ├── __init__.py
│   │
│   ├── __pycache__
│   │
│   ├── endpoints
│   │   │
│   │   ├── __init__.py
│   │   │
│   │   ├── __pycache__
│   │   │
│   │   ├── audio.py
│   │   │
│   │   ├── download.py
│   │   │
│   │   ├── frames.py
│   │   │
│   │   ├── health.py
│   │   │
│   │   ├── management.py
│   │   │
│   │   ├── ocr.py
│   │   │
│   │   ├── status.py
│   │   │
│   │   └── videos.py
│   │
│   └── routes.py
│
├── core
│   │
│   ├── __init__.py
│   │
│   ├── __pycache__
│   │
│   ├── auth.py
│   │
│   ├── aws.py
│   │
│   ├── config.py
│   │
│   ├── logging.py
│   │
│   ├── middleware.py
│   │
│   └── state.py
│
├── data
│   │
│   └── brand_database.json
│
├── main.py
│
├── models
│   │
│   ├── __init__.py
│   │
│   ├── __pycache__
│   │
│   ├── api.py
│   │
│   └── status_tracker.py
│
├── requirements.txt
│
├── services
│   │
│   ├── __init__.py
│   │
│   ├── __pycache__
│   │
│   ├── audio_processing.py
│   │
│   ├── frames_processing.py
│   │
│   ├── ocr_processing
│   │   │
│   │   ├── __init__.py
│   │   │
│   │   ├── __pycache__
│   │   │
│   │   ├── main_ocr_processing.py
│   │   │
│   │   ├── ocr_brand_matching.py
│   │   │
│   │   └── ocr_cleaning.py
│   │
│   ├── s3_operations.py
│   │
│   ├── status_processing.py
│   │
│   ├── video_post_processing.py
│   │
│   └── video_processing.py
│
├── tests
│   │
│   ├── __init__.py
│   │
│   ├── processing_optimisation_test.py
│   │
│   ├── test_upload.py
│   │
│   └── test_upload_local.py
│
└── utils
    │
    ├── __init__.py
    │
    ├── __pycache__
    │
    ├── decorators.py
    │
    └── utils.py
