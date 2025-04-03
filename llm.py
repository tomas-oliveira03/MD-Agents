from together import Together
import os
from dotenv import load_dotenv

def get_llm_response(prompt):
    # Carregar a chave da API
    load_dotenv()
    client = Together(api_key=os.getenv("API_KEY"))
    
    # Enviar para o modelo da Together e obter resposta
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
