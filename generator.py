from langchain_ollama import OllamaLLM
from retriever import RAGretriever
from db import vector_store
from embedding import embedding_manager

# initialize llm
llm = OllamaLLM(model="nemotron-3-super:cloud")


# Initialize retriever
rag_retriever = RAGretriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager
)


def rag(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):

    # Retrieve relevant documents
    results = retriever.retrieve(
        query,
        top_k=top_k,
        score_threshold=min_score
    )

    if not results:
        return {
            'answer': 'No relevant context found',
            'sources': [],
            'confidence': 0.0
        }

    # Build context from retrieved documents
    context = "\n\n".join(
        [doc["document"] for doc in results]
    )

    # Build source information
    sources = [
        {
            'source': doc['metadata'].get(
                'source_file',
                doc['metadata'].get('source', 'unknown')
            ),
            'page': doc['metadata'].get('page', 'unknown'),
            'similarity_score': doc['similarity_score'],
            'preview': doc['document'][:300] + '...'
        }
        for doc in results
    ]

    # Highest similarity score
    confidence = max(
        [doc['similarity_score'] for doc in results]
    )

    # Create prompt
    prompt = f"""
Use the following context to answer the question.

Answer only using the information provided in the context.
If the answer cannot be found in the context, say that you do not have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    output = {
        'answer': response,
        'sources': sources,
        'confidence': confidence
    }

    if return_context:
        output['context'] = context

    return output


# Ask a question
result = rag(
    "What is an LSTM?",
    rag_retriever,
    llm,
    top_k=3,
    min_score=0.1,
    return_context=True
)

print("Answer:", result['answer'])
print("Sources:", result['sources'])
#print("Confidence:", result['confidence'])
# print("Context Preview:", result['context'][:300])

