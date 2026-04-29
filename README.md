AI Document Intelligence - RAG System 📄🔍
Project Overview
This is a Retrieval-Augmented Generation (RAG) based system designed to search and extract precise information from multiple PDF documents (like HR Policies or Handbooks). Instead of manually reading through hundreds of pages, this tool finds the most relevant answers based on user queries.
How it Works
Document Loading: Uses PyPDF2 to read multiple PDF files simultaneously.
Preprocessing: Uses NLTK to clean the text (removing stop words and performing stemming).
Vectorization: Converts text into numerical data using TF-IDF Vectorizer.
Similarity Search: Calculates Cosine Similarity between the user's question and the document sentences to find the best matches.
Tech Stack
Language: Python
Libraries: NLTK, Scikit-learn, PyPDF2, NumPy
How to Run
Ensure Python is installed.
Install dependencies: pip install PyPDF2 nltk scikit-learn numpy
Add your PDF files to the specified directory.
Run the script: python microllm.ai.py
Type your question in the terminal to see the top matching answers.
