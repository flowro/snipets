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

############################################################
############################################################
############################################################

from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from datasets import load_dataset

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", passages=passages)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

def generate_summary(query):
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Assuming 'passages' is a collection of texts related to your applications
passages = [
    {"text": "Application App1 managed by Alice, high security risk.", "title": ""},
    {"text": "Application App2 managed by Bob, low security risk.", "title": ""}
]

query = "Tell me about App1."
summary = generate_summary(query)
print(summary)


##################################
###################################
#####################################

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Function to generate a formal paragraph from metadata
def generate_formal_paragraph(app_name, owner, risk_level, other_details):
    # Create a prompt with a formal tone request
    prompt = f"translate to formal English: Application Name: {app_name}. Owner: {owner}. Risk Level: {risk_level}. Details: {other_details}."
    
    # Encode the prompt and generate output
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=200)
    
    # Decode and print the summary
    formal_paragraph = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return formal_paragraph

# Example metadata
app_name = "App1"
owner = "Alice"
risk_level = "High"
other_details = "Used for managing client data and transactions."

# Generate and print the formal paragraph
formal_paragraph = generate_formal_paragraph(app_name, owner, risk_level, other_details)
print(formal_paragraph)

