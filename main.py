import os
from llm import get_llm_response
from embeddings import load_papers, create_embeddings, retrieve_relevant_chunks
import initPrompt

def main():
    
    # Definir o caminho dos arquivos dos papers
    paper_paths = [f"papers/paper{i}.txt" for i in range(1, 6)]
    
    # Carregar os papers e criar os embeddings
    papers = load_papers(paper_paths)
    chunks = create_embeddings(papers)
    
    # Pergunta do usuário
    prompt = "Quantas horas devo dormir por dia?"
    
    # Recuperar os chunks mais relevantes
    relevant_chunks = retrieve_relevant_chunks(prompt, chunks, top_k=5)
    
    # Formatar o contexto
    context = "\n\n".join([f"Paper {c['paper_id']}:\n{c['content']}" for c in relevant_chunks])
    
    # Montar o prompt final para o LLM
    final_prompt = f"""
    {initPrompt.initial_prompt}

    ### Trechos dos papers:
    {context}

    ### Pergunta:
    {prompt}
    """
    
    print("Contexto RAG:")
    print(context)
    print("########")
    print("########")
    print("########\n")
    
    # Obter a resposta do LLM
    response = get_llm_response(final_prompt)
    
    # Exibir a resposta gerada
    print("Resposta do LLM:")
    print(response)
    

if __name__ == "__main__":
    main()
