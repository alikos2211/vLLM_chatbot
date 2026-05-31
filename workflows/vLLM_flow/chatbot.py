import os
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class GPT2Chatbot:
    def __init__(self, max_history=6):

        self.client = OpenAI(
            base_url="http://localhost:8000/v1",  
            api_key="not needed"
        )

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX"))

        self.model_emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

        self.history = []
        self.max_history = max_history
        self.system_prompt = (
            "The following is a conversation with a helpful AI assistant.\n"
            "User: Hello\n"
            "Bot: Hi! How can I help you today?\n"
        )
        

    def build_prompt(self):
        prompt = self.system_prompt
        for turn in self.history[-self.max_history:]:  # rolling window
            prompt += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        return prompt

    def chat(self, user_input):

        # Retrieve from Pinecone first
        query_vector = self.model_emb.encode([user_input])[0].tolist()
        results = self.index.query(vector=query_vector, top_k=5, include_metadata=True)

        if results["matches"]:
           relevant_chunks = [match["metadata"]["text"] for match in results["matches"]]
           retrieved_context = " ".join(relevant_chunks)
           print("Context IDs:", [match["id"] for match in results["matches"]])
        else:
           retrieved_context = "No relevant context found."

        prompt = self.build_prompt()
        prompt += f"Context: {retrieved_context}\nUser: {user_input}\nBot:"


        try:
            
            response = self.client.completions.create(
                model="gpt2",
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                stop=["User:", "\n\n"],
                
            )
            bot_reply = response.choices[0].text.strip()
        except Exception as e:
            return f"[Error: {e}]"

        if not bot_reply:
            return "[No response — try rephrasing]"

        self.history.append({"user": user_input, "bot": bot_reply})
        return bot_reply

    def clear_history(self):
        self.history = []


