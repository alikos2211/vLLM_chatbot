import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
model.to(device)



class GPT2Chatbot:
    def __init__(self, max_history=6):

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
        for turn in self.history[-self.max_history:]:
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
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            new_tokens = output_ids[0][input_len:]
            bot_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Replicate the stop sequences from the original
            for stop in ["User:", "\n\n"]:
                if stop in bot_reply:
                    bot_reply = bot_reply[:bot_reply.index(stop)].strip()

        except Exception as e:
            return f"[Error: {e}]"

        if not bot_reply:
            return "[No response — try rephrasing]"

        self.history.append({"user": user_input, "bot": bot_reply})
        return bot_reply

    def clear_history(self):
        self.history = []


