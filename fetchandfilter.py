import os
import base64
import json
import pandas as pd
import joblib
import nltk
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords for NLP processing
nltk.download('stopwords')

# Gmail API Scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Paths for AI model and training data
MODEL_PATH = "job_email_classifier.pkl"
TRAINING_DATA_PATH = "training_data.csv"

# Train AI Model Using Real Data
def train_email_classifier():
    """Retrain AI model using balanced job & spam emails."""
    
    if not os.path.exists(TRAINING_DATA_PATH):
        print("⚠️ No real training data found! Using default dataset.")
        return

    df = pd.read_csv(TRAINING_DATA_PATH)

    # Balance dataset (equal job & spam emails)
    job_emails = df[df["label"] == "job"]
    spam_emails = df[df["label"] == "spam"]
    
    num_jobs = len(job_emails)
    if num_jobs == 0 or len(spam_emails) == 0:
        print("❌ Error: Training data must have both 'job' and 'spam' categories.")
        return
    
    spam_sample = spam_emails.sample(n=min(num_jobs, len(spam_emails)), random_state=42)
    df_balanced = pd.concat([job_emails, spam_sample]).sample(frac=1, random_state=42)

    # Prepare training data
    training_texts = (df_balanced["subject"] + " " + df_balanced["sender"]).tolist()
    training_labels = [1 if label == "job" else 0 for label in df_balanced["label"]]

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(training_texts)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, training_labels)

    # Save trained model
    joblib.dump((model, vectorizer), MODEL_PATH)
    print(f"✅ AI Model Retrained with {num_jobs} Job Emails & {num_jobs} Spam Emails (Balanced).")

# Authenticate Gmail API
def authenticate_gmail():
    """Authenticate and connect to Gmail API."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds

# Fetch Emails from Gmail "Jobs" Label
def fetch_jobs_label_emails():
    """Fetch all emails from Gmail 'Jobs' label and include sender details."""
    
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)
    query = "label:Jobs"
    results = service.users().messages().list(userId="me", q=query, maxResults=200).execute()
    messages = results.get("messages", [])

    job_emails = []
    for msg in messages:
        msg_id = msg["id"]
        email = service.users().messages().get(userId="me", id=msg_id).execute()
        headers = email["payload"]["headers"]

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")

        job_emails.append({"subject": subject, "sender": sender, "label": "job"})

    return job_emails

# Fetch Spam Emails
def fetch_spam_emails():
    """Fetch spam emails from Gmail categories (Social, Forums, Promotions)."""
    
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)
    categories = {"category:social": "Social", "category:forums": "Forums", "category:promotions": "Promotions"}

    spam_emails = []
    for category, label in categories.items():
        print(f"\n📩 Fetching {label} emails...")
        results = service.users().messages().list(userId="me", q=category, maxResults=100).execute()
        messages = results.get("messages", [])

        for msg in messages:
            msg_id = msg["id"]
            email = service.users().messages().get(userId="me", id=msg_id).execute()
            headers = email["payload"]["headers"]

            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
            sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")

            spam_emails.append({"subject": subject, "sender": sender, "label": "spam"})

    return spam_emails

# Classify Emails Using AI
def classify_emails_with_ml(emails):
    """Classify emails using AI with subject and sender information."""

    model, vectorizer = joblib.load(MODEL_PATH)
    job_emails = []
    spam_keywords = ["instagram", "linkedin messaging", "duolingo", "facebook", "snapchat", "newsletter", "promotion"]
    job_domains = ["smartrecruiters.com", "successfactors.eu", "bosch.com", "dlr.de", "softgarden.io"]

    for email in emails:
        subject_lower = email["subject"].lower()
        sender_lower = email["sender"].lower()

        if any(domain in sender_lower for domain in job_domains):
            print(f"✅ Keeping (Known Job Domain): {email['subject']} from {email['sender']}")
            job_emails.append(email)
            continue

        if any(kw in subject_lower for kw in spam_keywords) or any(kw in sender_lower for kw in spam_keywords):
            print(f"🛑 Skipping (Detected as Social Media/Spam): {email['subject']} from {email['sender']}")
            continue

        email_text = email["subject"] + " " + email["sender"]
        X_test = vectorizer.transform([email_text])
        prediction = model.predict(X_test)[0]

        if prediction == 1:
            job_emails.append(email)

    return job_emails

# Run the script
if __name__ == "__main__":
    print("\n📩 Fetching job & spam emails and updating training data...")
    job_emails = fetch_jobs_label_emails()
    spam_emails = fetch_spam_emails()
    
    df_existing = pd.DataFrame(columns=["subject", "sender", "label"])
    df_combined = pd.concat([df_existing, pd.DataFrame(job_emails), pd.DataFrame(spam_emails)]).drop_duplicates()
    df_combined.to_csv(TRAINING_DATA_PATH, index=False)

    train_email_classifier()
    print("\n📩 Fetching new emails...")
    emails = fetch_jobs_label_emails()
    print("\n🔍 Filtering job-related emails using AI...")
    job_emails = classify_emails_with_ml(emails)

    pd.DataFrame(job_emails).to_csv("filtered_job_emails.csv", index=False)
    print("\n✅ Filtered job emails saved to 'filtered_job_emails.csv'!")
