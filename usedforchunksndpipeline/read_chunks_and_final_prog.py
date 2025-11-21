import requests
import os 
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PARQUET_FILE = "embeddings.parquet"

def create_embedding(text_list ):
        r= requests.post("http://127.0.0.1:11434/api/embed" , json ={
            "model" : "bge-m3",
            "input" : text_list
        })

        embedding = r.json()["embeddings"]
        return embedding


if os.path.exists(PARQUET_FILE):

   # print("Parquet file already exists. Loading directly...")
    df = pd.read_parquet(PARQUET_FILE)
    
else:

    jsons = os.listdir("jsons")
    my_dicts=[]
    chunk_id=0
    x= 1
    for json_file in jsons:
        with open (f"jsons/{json_file}") as f:
            content = json.load(f)
        print(f"working on {x}")
        x+=1
        embeddings = create_embedding([c['text'] for c in content['chunks']]) # a list that contains all text from chunks 
        for i, chunk in enumerate(content['chunks']):

            chunk['chunk_id'] = chunk_id
            chunk["embedding"]= embeddings[i]
            chunk_id+=1
            my_dicts.append(chunk)
    
        df = pd.DataFrame.from_records(my_dicts)


       
#print(df)
df.to_parquet(PARQUET_FILE)
#print("\nSaved as embeddings.parquet successfully!")

incoming_query = input("Ask a question : ")
question_embedding = create_embedding([incoming_query])[0]

similarity = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
top_results = 5
max_index= similarity.argsort()[::-1][0:top_results]
new_df= df.loc[max_index]
print(new_df[["title", "number", "text"]])


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response



prompt = f'''I am teaching web development in my Sigma web development course. 
Here are video subtitle chunks containing video title, video number, 
start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]]\
                   .reset_index(drop=True)\
                   .to_json(orient="records")}

---------------------------------
"{incoming_query}"
User asked this question related to the video chunks,
 you have to answer in a human way (dont mention the above format, 
 its just for you) where and how much content is taught in which
   video (in which video and at what timestamp) and 
   guide the user to go to that particular video. 
   If user asks unrelated question, tell him that you 
   can only answer questions related to the course
'''


with open("prompt.txt","w") as f:
     f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)