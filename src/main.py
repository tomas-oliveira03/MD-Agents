import pickle  # For saving and loading embeddings
from llm import get_llm_response
from embeddings import load_papers, create_embeddings, retrieve_relevant_chunks
import initPrompt
import os

def main():
    
    # Define paths for the papers and embeddings
    paper_paths = [f"papers/paper{i}.txt" for i in range(1, 6)]
    embeddings_file = "embeddings/embeddings.pkl" 
    
    # Check if the embeddings file exists
    if os.path.exists(embeddings_file):
        print("Loading existing embeddings...")
        with open(embeddings_file, "rb") as f:
            chunks = pickle.load(f)
            
    else:
        print("Generating new embeddings...")
        papers = load_papers(paper_paths)
        chunks = create_embeddings(papers)
        
        # Save the embeddings to a file
        with open(embeddings_file, "wb") as f:
            pickle.dump(chunks, f)
        print("Embeddings saved to file.")
    
    

    while True:
        # Pergunta do usuário
        prompt = input("Digite a sua pergunta (ou 'EXIT' para sair): \n>>>")
        if prompt.strip().upper() == "EXIT":
            break
        
        # Recuperar os chunks mais relevantes
        relevant_chunks = retrieve_relevant_chunks(prompt, chunks, top_k=5)
        
        # Formatar o contexto
        context = "\n\n".join([f"Paper {c['paper_id']}:\n{c['content']}" for c in relevant_chunks])
        
        # Montar o prompt final para o LLM
        final_prompt = f""" {initPrompt.initial_prompt}

### Pergunta:
{prompt}

### Contexto de artigos:
{context}
"""
        
        print("\nContexto RAG:")
        print(context)
        print("########")
        print("########")
        print("########\n")
        
        # Obter a resposta do LLM
        response = get_llm_response(final_prompt)

        print("Resposta do LLM:")
        print(response)

if __name__ == "__main__":
    main()
