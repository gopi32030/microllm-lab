# 🧠 MicroLLM Lab: Custom Neural Network from Scratch

### 📌 Project Overview
MicroLLM Lab is an entirely custom-built, offline Generative AI training environment. Instead of relying on heavy frameworks like PyTorch or TensorFlow, this project builds a core Neural Network Architecture *from scratch using pure NumPy matrix multiplications*. It features a Streamlit UI where users can train their own micro language model on custom text data and visualize the loss convergence in real-time.

---

### 🚀 Key Features & Capabilities
* *Custom Tokenizer:* Built-in vocabulary generation mapping words to unique IDs, handling <PAD>, <UNK>, <START>, and <END> tokens.
* *Pure Math Architecture:* Forward and backward propagation, He initialization, and momentum-based optimization—all implemented via raw NumPy.
* *Advanced Sampling Controls:* Includes Temperature, Top-K, and Top-P (Nucleus) sampling, along with repetition penalties for high-quality text generation.
* *Interactive UI:* Streamlit-powered dashboard for real-time training, hyperparameter tuning (Learning Rate, Hidden Size, Epochs), and visual loss analytics.

---

### 📸 Application Walkthrough & Visual Analytics
Behold the dashboard interface showcasing custom training convergence, interactive analytics, and precise vocabulary predictions:

#### 1. Real-Time Model Training & Loss Convergence <img width="1366" height="720" alt="Screenshot 2026-07-14 211325" src="https://github.com/user-attachments/assets/cad010d0-82a2-4b99-ad66-644206a102cf" />
#### 2. Advanced Sampling Controls & Custom Generation <img width="1366" height="720" alt="Screenshot 2026-07-14 211545" src="https://github.com/user-attachments/assets/6b53eada-cb49-4bcb-9580-00b08c85568d" />
#### 3. Deep Data Metrics & Vocabulary Distribution Chart <img width="1366" height="720" alt="Screenshot 2026-07-14 212358" src="https://github.com/user-attachments/assets/8f73461b-9db4-4e4d-8a01-9754047b7a1f" />
#### 4. System Integrity & System Diagnostics <img width="1366" height="720" alt="Screenshot 2026-07-14 212516" src="https://github.com/user-attachments/assets/25b20e78-e6b3-41ef-b086-5283edb9630d" />

---

### 🛠️ Tech Stack & Libraries
* *Core Logic:* Python, NumPy
* *Visualization:* Matplotlib
* *User Interface:* Streamlit

---

  ### 💻 How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gopi32030/microllm-lab.git
   cd microllm-lab

2.Install Dependencies : pip install -r requirements.txt

3.launch the application : streamlit run microllm.ai.py
