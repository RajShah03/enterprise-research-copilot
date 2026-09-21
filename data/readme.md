@"
# Data Directory

Place the source research documents used by the application in this directory.

The actual source PDFs are intentionally excluded from Git because redistribution
rights should be verified before publishing third-party documents.

After adding documents locally, run:

python ingest.py
"@ | Set-Content "data\README.md" -Encoding UTF8