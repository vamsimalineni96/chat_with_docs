from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI


def run(db_schema):
    client = OpenAI(
    base_url = os.getenv("NVIDIA_LLM_ENDPOINT"),
    api_key = os.getenv("NVIDIA_NIM_API_KEY")
    )

    prompt=f"""
    You are given the complete schema of a SQLite database.
    Generate a precise, factual summary describing the purpose and contents of the database.
    The summary must be strictly grounded in the schema—do NOT invent information.

    Schema:
    {db_schema}

    Instructions:
    1. Identify the main domain of the database (e.g., sports, retail, education, etc.).
    2. List the main entities represented by the tables.
    3. Describe the key relationships between entities (foreign keys, one-to-many, etc.).
    4. Highlight important attributes that define each entity (important columns).
    5. Keep the summary concise (150–250 words) but detailed enough so a language model
    could choose this database when given a natural-language question.

    Output Format:
    a valid json object
    Domain: <one phrase>
    Summary:
    <one or two detailed paragraphs summarizing the database contents and relationships>
    
    Donot return ```json ```
    """


    completion = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role":"user","content":prompt}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=False
    )

    return completion.choices[0].message.content


