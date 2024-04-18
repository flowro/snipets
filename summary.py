from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to generate summaries
def generate_summary(app_name, owner, risk_level):
    description = f"The application '{app_name}' is owned by {owner}. It has a risk level described as {risk_level}."
    summary = summarizer(description, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Example metadata
applications = [
    {"name": "App1", "owner": "Alice", "risk": "High"},
    {"name": "App2", "owner": "Bob", "risk": "Low"},
]

# Generate and print summaries
for app in applications:
    summary = generate_summary(app['name'], app['owner'], app['risk'])
    print(f"Summary for {app['name']}: {summary}")
