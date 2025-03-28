import os
import pandas as pd
import time
import random
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from tqdm import tqdm  # Progress bar

# ✅ Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEYV")

if not groq_api_key:
    raise ValueError("🚨 ERROR: Missing Groq API Key! Please check your .env file.")

# ✅ Initialize Groq's LLM with a timeout
MODEL_NAME = "gemma2-9b-it"  # Define model name explicitly
chat_model = ChatGroq(model_name=MODEL_NAME, groq_api_key=groq_api_key, timeout=30)

# ✅ Load CSV file
file_path = "C:/Chatbot/Raw Sheets/Thousand6.csv"
df = pd.read_csv(file_path,encoding="iso-8859-1")


# ✅ Function to extract details including Hotel Code
def extract_chat_details(conversation, max_retries=3):
    """Extracts summary, asking phrase, solution, and hotel code from a chat transcript using LLM."""


    if pd.isna(conversation) or not isinstance(conversation, str) or conversation.strip() == "":
        return "", "", "", ""  # Skip empty chats

    prompt = f"""
    Extract the following details from the chat transcript:
    - Chat Summary: A short summary of the entire conversation.
    - Asking Phrase: The main question asked by the visitor.
    - Solution Provided: The response given by the agent.
    - Hotel Code: If the conversation mentions a specific hotel code, extract it. Otherwise, return 'N/A'.
    
    Chat Transcript:
    {conversation}
    
    Return the result in this format:
    Chat Summary: <summary>
    Asking Phrase: <question>
    Solution Provided: <solution>
    Hotel Code: <hotel_code>
    """

    retry_delay = 2  # Start with 2 seconds
    for attempt in range(max_retries):
        try:
            response = chat_model.invoke(
                [
                    SystemMessage(content="You are an AI assistant extracting key details from customer chats."),
                    HumanMessage(content=prompt)
                ],
                timeout=30  # Timeout after 30s
            )

            extracted_text = response.content.strip()

            summary, question, solution, hotel_code = "", "", "", "N/A"
            for line in extracted_text.split("\n"):
                if line.startswith("Chat Summary:"):
                    summary = line.replace("Chat Summary:", "").strip()
                elif line.startswith("Asking Phrase:"):
                    question = line.replace("Asking Phrase:", "").strip()
                elif line.startswith("Solution Provided:"):
                    solution = line.replace("Solution Provided:", "").strip()
                elif line.startswith("Hotel Code:"):
                    hotel_code = line.replace("Hotel Code:", "").strip()

            return summary, question, solution, hotel_code

        except Exception as e:
            print(f"⚠️ Error on attempt {attempt + 1}: {e}")
            time.sleep(retry_delay)  # Exponential backoff
            retry_delay *= random.uniform(1.5, 2.5)  # Randomize retry delay

    return "", "", "", "N/A"  # Return empty if all retries fail

# ✅ Process chats in batches (to avoid API overload)
batch_size = 10  # Adjust batch size as needed
extracted_data = []

for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches", unit="batch"):
    batch = df.iloc[i : i + batch_size]
    batch_results = batch["Conversation"].apply(lambda x: pd.Series(extract_chat_details(x)))
    extracted_data.extend(batch_results.values.tolist())  # Convert DataFrame to list

# ✅ Append extracted data to DataFrame
df[["Chat Summary", "Asking Phrase", "Solution Provided", "Hotel Code"]] = pd.DataFrame(extracted_data, index=df.index)

# ✅ Keep only required columns
df = df[["Chat ID", "Chat Duration", "Chat Summary", "Asking Phrase", "Solution Provided", "Hotel Code"]]

# ✅ Save structured data
output_file = "C:/Chatbot/Processed Sheets/Structured_Chat_split1_7.csv"
df.to_csv(output_file, index=False)

print(f"✅ Data structured and saved as '{output_file}'")

# ✅ Display sample output
print(df.head())
