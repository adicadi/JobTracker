import os
import base64
import pandas as pd
import json
import re
import nltk
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Download stopwords for NLP processing
nltk.download('stopwords')

# Define the scope (read-only access to Gmail)
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# German and English Job-Related Keywords
JOB_KEYWORDS = [
    "job application", "interview", "hiring", "recruiter", "career opportunity", "resume", "CV", "HR",
    "internship", "shortlisted", "final round", "work student", "mandatory internship", "recruitment", 
    "job position", "open position", "career", "job offer", "apply now", "candidate",
    "bewerbung", "bewerbungsunterlagen", "lebenslauf", "vorstellungsgespräch", "bewerbungsgespräch",
    "praktikum", "werkstudent", "jobangebot", "karriere", "personalabteilung", "stellenausschreibung",
    "bewerbungsprozess", "jobchance", "jobportal", "einstellungsprozess", "neue karrierechance"
]

# Load Exclude List from Config File
def load_exclusion_lists():
    """Load excluded senders and keywords dynamically from config.json"""
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["exclude_senders"], config["exclude_keywords"]

# Use the dynamic lists in filtering
EXCLUDE_SENDERS, EXCLUDE_KEYWORDS = load_exclusion_lists()

def authenticate_gmail():
    """Authenticate and connect to Gmail API."""
    creds = None

    # Load existing token or authenticate
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the access token for next time
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds

def fetch_emails():
    """Fetches emails from Gmail API."""
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)

    # Get the user's inbox messages (fetching latest 30 emails)
    results = service.users().messages().list(userId="me", maxResults=30).execute()
    messages = results.get("messages", [])

    emails = []

    for msg in messages:
        msg_id = msg["id"]
        email = service.users().messages().get(userId="me", id=msg_id).execute()

        payload = email["payload"]
        headers = payload["headers"]

        # Extract subject and sender
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")

        # Extract email body
        body = ""
        if "parts" in payload:
            for part in payload["parts"]:
                if part["mimeType"] == "text/plain":
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                    break
        else:
            body = "No body available"

        emails.append({"sender": sender, "subject": subject, "body": body})

    return emails

def classify_emails(emails):
    """Classifies emails as job-related or not using Logistic Regression model."""

    # Sample training data (job-related vs. non-job emails)
    training_texts = [
        "Job application received for software engineer",
        "Your interview has been scheduled",
        "Hiring for a new data scientist role",
        "Apply now for an open internship position",
        "We are interested in your resume",
        "Discount on your favorite shoes!",
        "Your flight ticket booking is confirmed",
        "Newsletter: The latest fashion trends",
        "Exclusive offer: 30% off on electronics",
        "Reminder: Your gym membership renewal is due"
    ]
    
    training_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 = Job-related, 0 = Not job-related

    # Vectorization with stopword removal (English & German)
    stop_words = stopwords.words('english') + stopwords.words('german')
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X_train = vectorizer.fit_transform(training_texts)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, training_labels)

    # Classify new emails
    job_emails = []
    email_texts = [email["subject"] + " " + email["body"] for email in emails]
    X_test = vectorizer.transform(email_texts)
    predictions = model.predict(X_test)

    for i, email in enumerate(emails):
        if predictions[i] == 1 and not any(kw in email["subject"].lower() for kw in EXCLUDE_KEYWORDS):
            job_emails.append(email)

    return job_emails

def save_to_csv(job_emails):
    """Saves filtered job emails to a CSV file."""
    df = pd.DataFrame(job_emails)
    df.to_csv("filtered_job_emails.csv", index=False)
    print("\n✅ Filtered job emails saved to 'filtered_job_emails.csv'!")

# Run the script
if __name__ == "__main__":
    print("\n📩 Fetching emails...")
    emails = fetch_emails()

    print("\n🔍 Filtering job-related emails...")
    job_emails = classify_emails(emails)

    if job_emails:
        for mail in job_emails:
            print(f"\n📧 {mail['subject']} - {mail['sender']}")
        save_to_csv(job_emails)
    else:
        print("\n✅ No job-related emails found!")
