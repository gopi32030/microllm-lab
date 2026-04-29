# ==========================================================
# MICROLLM LAB - IMPROVED VERSION v2.0 (STABLE)
# Better Training | Better Architecture | Production Ready
# Fixed for Gopi - No Slider Log Error
# ==========================================================

import numpy as np
import random
import streamlit as st
import os
import pickle
import json
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# To avoid unnecessary usage stat popups
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="MicroLLM Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    .success-box {
        background: rgba(76, 175, 80, 0.2);
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ================== TOKENIZER ==================
class ImprovedTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def build_vocab(self, text, min_freq=1):
        tokens = text.lower().split()
        freq = Counter(tokens)
        vocab = sorted([w for w, f in freq.items() if f >= min_freq])
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        vocab = special_tokens + vocab
        self.word_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.vocab_size = len(vocab)
        return self.vocab_size

    def encode(self, text):
        tokens = text.lower().split()
        return [self.word_to_id.get(w, self.word_to_id['<UNK>']) for w in tokens]

    def decode(self, ids):
        return " ".join([self.id_to_word.get(i, '<UNK>') for i in ids])


# ================== NEURAL NETWORK ==================
class MicroLLM:
    def __init__(self, vocab_size, hidden_size=64, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.history = []

        # He initialization
        self.W1 = np.random.randn(vocab_size, hidden_size) * np.sqrt(2.0 / vocab_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, vocab_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, vocab_size))
        self.vW2 = np.zeros_like(self.W2)

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, token_id):
        x = np.zeros((1, self.vocab_size))
        x[0, token_id] = 1.0
        hidden = np.tanh(np.dot(x, self.W1) + self.b1)
        logits = np.dot(hidden, self.W2) + self.b2
        probs = self.softmax(logits)
        return probs[0], hidden[0]

    def backward(self, token_id, next_token_id, hidden):
        probs, _ = self.forward(token_id)
        dlogits = probs.copy()
        dlogits[next_token_id] -= 1.0
        dW2 = np.outer(hidden, dlogits)
        db2 = dlogits
        momentum = 0.9
        self.vW2 = momentum * self.vW2 - self.learning_rate * dW2
        self.W2 += self.vW2
        self.b2 -= self.learning_rate * db2

    def train_step(self, token_id, next_token_id):
        probs, hidden = self.forward(token_id)
        loss = -np.log(probs[next_token_id] + 1e-9)
        self.backward(token_id, next_token_id, hidden)
        return loss


# ================== SAMPLING CONTROLS ==================
def apply_controls(probs, temp, k, p, generated_ids, penalty):
    # Temperature
    if temp > 0:
        probs = np.exp(np.log(probs + 1e-9) / temp)
        probs /= probs.sum()

    # Repetition Penalty
    for token_id in set(generated_ids):
        probs[token_id] /= penalty
    probs /= probs.sum()

    # Top-K
    if k > 0:
        idx = np.argsort(probs)[-k:]
        new_probs = np.zeros_like(probs)
        new_probs[idx] = probs[idx]
        probs = new_probs / new_probs.sum()

    # Top-P
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = cumulative <= p
    cutoff[0] = True
    mask = sorted_idx[cutoff]
    new_probs = np.zeros_like(probs)
    new_probs[mask] = probs[mask]
    return new_probs / new_probs.sum()


# ================== STREAMLIT UI ==================
st.markdown("""
<div class="main-header">
    <h1>🧠 MicroLLM Lab</h1>
    <p><i>Offline Generative AI Training Environment</i></p>
</div>
""", unsafe_allow_html=True)

if 'model' not in st.session_state: st.session_state.model = None
if 'tokenizer' not in st.session_state: st.session_state.tokenizer = None
if 'training_data' not in st.session_state: st.session_state.training_data = None

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Train", "📝 Generate", "📊 Analytics", "ℹ️ About"])

# ------------ TAB 1: TRAINING ------------
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Training Data")
        training_text = st.text_area("Input Text:", value="hello how are you\ni am fine\nlearning ai is fun",
                                     height=150)
    with col2:
        st.subheader("Settings")
        h_size = st.slider("Hidden Size", 16, 256, 64)
        # FIXED: Removed log=True for compatibility
        l_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
        epochs = st.slider("Epochs", 5, 100, 20)

    if st.button("🔥 Start Training", use_container_width=True):
        if not training_text.strip():
            st.error("Text empty!")
        else:
            tokenizer = ImprovedTokenizer()
            v_size = tokenizer.build_vocab(training_text)
            model = MicroLLM(v_size, h_size, l_rate)

            tokens = training_text.lower().split()
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

            losses = []
            prog = st.progress(0)
            for e in range(epochs):
                e_loss = 0
                random.shuffle(pairs)
                for t1, t2 in pairs:
                    e_loss += model.train_step(tokenizer.word_to_id[t1], tokenizer.word_to_id[t2])
                avg_l = e_loss / len(pairs)
                losses.append(avg_l)
                prog.progress((e + 1) / epochs)

            st.session_state.model, st.session_state.tokenizer = model, tokenizer
            st.session_state.training_data = {'text': training_text, 'v_size': v_size, 'losses': losses}
            st.success("✅ Training Done!")

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(losses, color='#667eea')
            ax.set_title("Loss Curve")
            st.pyplot(fig)

# ------------ TAB 2: GENERATION ------------
with tab2:
    if st.session_state.model:
        c1, c2 = st.columns([2, 1])
        with c1:
            prompt = st.text_input("Prompt:", "hello")
            max_t = st.slider("Length", 5, 50, 15)
        with c2:
            temp = st.slider("Temp", 0.1, 1.5, 0.7)
            tk = st.slider("Top-K", 0, 10, 3)

        if st.button("✨ Generate"):
            ids = st.session_state.tokenizer.encode(prompt)
            if not ids: ids = [0]

            for _ in range(max_t):
                probs, _ = st.session_state.model.forward(ids[-1])
                probs = apply_controls(probs, temp, tk, 0.9, ids, 1.2)
                next_id = np.random.choice(len(probs), p=probs)
                ids.append(next_id)
                if st.session_state.tokenizer.id_to_word[next_id] == '<END>': break

            result = st.session_state.tokenizer.decode(ids)
            st.markdown(f'<div class="success-box"><b>Result:</b> {result}</div>', unsafe_allow_html=True)
    else:
        st.info("Train the model first!")

# ------------ TAB 3 & 4: ANALYTICS & ABOUT ------------
with tab3:
    if st.session_state.training_data:
        data = st.session_state.training_data
        st.metric("Vocab Size", data['v_size'])
        words = data['text'].lower().split()
        top_10 = Counter(words).most_common(10)
        names, counts = zip(*top_10)
        fig, ax = plt.subplots()
        ax.bar(names, counts, color='#764ba2')
        st.pyplot(fig)

with tab4:
    st.write("### 👨‍💻 Developed by Gopi")
    st.write("MicroLLM is a custom NumPy-based neural network for learning LLM basics.")