import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_pdf(file_path: str):
    pdf = pymupdf.open(file_path)
    
    full_text =  ""
    for page in pdf:
        full_text += page.get_text()  # type: ignore

    pdf.close()
    return full_text

def split_text(text: str):
    split_text = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = split_text.split_text(text)
    return chunks

def load_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embeddings

def store_data(chunks: list, embeddings_model):
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings_model,
        persist_directory="Chroma_DB"
    )
    return vectorstore

#testing only
if __name__ == "__main__":
    text = load_pdf("upload/PDF-Test.pdf")
    chunks = split_text(text)

    e_model = load_embeddings()
    VectorStore = store_data(chunks, e_model)

    question = "Question?"
    result = VectorStore.similarity_search(question, k=3)
    
    print(f"top 3 most relevant chunk for '{question}'")
    for i, doc in enumerate(result):
        print(f"{i+1}. {doc.page_content}")