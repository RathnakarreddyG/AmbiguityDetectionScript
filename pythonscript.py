import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Load the model
# model = SentenceTransformer('all-MiniLM-L6-v2')

model = SentenceTransformer('all-mpnet-base-v2')

tqdm.pandas()

print("Model loaded successfully.")
# File paths
input_file = "input.xlsx"  # replace with your filename
output_file = "output_with_expected_intents.xlsx"

# Threshold
MATCH_THRESHOLD = 0.60

# Load the sheets
utterance_df = pd.read_excel(input_file, sheet_name='utterence_intent')
faq_df = pd.read_excel(input_file, sheet_name='FAQs')
print("Data loaded successfully.")
# Ensure columns
# utterance_df should have: ['Input', 'Intent']
# faq_df should have: ['Question', 'Intent']

# Function to find matching intent
def find_expected_intent(input_text, original_intent):
    # Step 1: If input and intent are semantically similar → keep same intent
    embeddings = model.encode([input_text, original_intent], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    if similarity >= MATCH_THRESHOLD:
        return original_intent

    # Step 2: Try matching with FAQ questions
    faq_questions = faq_df['Intent'].astype(str).tolist()
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embedding, faq_embeddings)[0]
    
    max_score_idx = cosine_scores.argmax().item()
    max_score = cosine_scores[max_score_idx].item()

    if max_score >= MATCH_THRESHOLD:
        return faq_df.iloc[max_score_idx]['Intent']
    
    # Step 3: No match
    return "none"

# Apply function
utterance_df["Expected Intent"] = utterance_df.progress_apply(
    lambda row: find_expected_intent(str(row["Input"]), str(row["Intent"])), axis=1
)

# Save output
utterance_df.to_excel(output_file, index=False)
print(f"Done. Output saved to {output_file}")
