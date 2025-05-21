import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the model (you can choose other models like 'all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load input Excel
input_file = "input.xlsx"  # replace with your file name
output_file = "output_with_ambiguity_check.xlsx"
df = pd.read_excel(input_file)

# Thresholds
MATCH_THRESHOLD = 0.75  # Above this = strong match
AMBIGUOUS_THRESHOLD = 0.55  # In-between = ambiguous

# Function to determine ambiguity
def get_ambiguity(user_query, faq_question):
    embeddings = model.encode([user_query, faq_question], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    if similarity > MATCH_THRESHOLD:
        return "Yes"  # Clear match, not ambiguous
    elif similarity > AMBIGUOUS_THRESHOLD:
        return "Yes"  # Likely ambiguous
    else:
        return "No"  # Not matching or unrelated

# Apply logic
df["Ambiguous"] = df.apply(lambda row: get_ambiguity(str(row[0]), str(row[2])), axis=1)

# Save output
df.to_excel(output_file, index=False)
print(f"Done. Output saved to {output_file}")
