import os
from dotenv import load_dotenv
import json
from pinecone import Pinecone, ServerlessSpec
import openai

load_dotenv()

pc = Pinecone(
    api_key="Your_Key",  
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
index_name = "Your_index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

openai.api_key = os.getenv("OPENAI_API_KEY") or "Your_Key" 

def generate_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def process_and_store_data(data):
    user_id = data['user']['id']
    
    for flight in data['user']['flights']:
        ticket_id = flight['ticket_id']
        
        journey_text = f"Flight from {flight['source']} to {flight['destination']} departs on {flight['departure_date']} and arrives on {flight['arrival_date']} with a layover of {flight['layover_duration']}."
        embedding = generate_embedding(journey_text)
        index.upsert([(str(ticket_id), embedding, {"text": journey_text})])
        
        for segment in flight['segments']:
            segment_text = f"{segment['flight_number']} from {segment['departure']['airport']} ({segment['departure']['iata']}) to {segment['arrival']['airport']} ({segment['arrival']['iata']}) departs on {segment['departure']['date']} and arrives on {segment['arrival']['date']}."
            embedding = generate_embedding(segment_text)
            index.upsert([(f"{ticket_id}-{segment['flight_number']}", embedding, {"text": segment_text})])
            
            for passenger in segment['passengers']:
                passenger_text = f"{passenger['first_name']} {passenger['last_name']} is seated in {passenger['seat_number']} with {passenger['cabin_baggage']} cabin baggage and {passenger['check_in_baggage']} checked baggage."
                embedding = generate_embedding(passenger_text)
                index.upsert([(f"{ticket_id}-{segment['flight_number']}-{passenger['seat_number']}", embedding, {"text": passenger_text})])


def query_and_generate_response(query):
    query_embedding = generate_embedding(query)
    
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    retrieved_info = "\n".join([result['metadata']['text'] for result in results['matches']])
    
    prompt = f"User Query: {query}\nRelevant Information:\n{retrieved_info}\nProvide a user-friendly response based on the above information with short words."

    print(prompt)

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message['content'].strip()


if __name__ == "__main__":
    data = load_data("/content/Journey_Details.json")  
    process_and_store_data(data)
    print("Data ingested successfully!")
    
    user_query = "What’s my seat for the first flight?"
    bot_response = query_and_generate_response(user_query)
    print(f"Bot Response: {bot_response}")