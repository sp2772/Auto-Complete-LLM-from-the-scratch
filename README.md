
# Auto-Complete LLM from Scratch

This repository demonstrates the step-by-step construction, training, and refinement of a **GPT-style Language Model** for auto-completion tasks, along with tools for data preprocessing, CLI interaction, and chatbot deployment.

Using the GPT multi Head attention architecture. Trained on openworldcorpus 

---

## 📁 Repository Structure

Auto_Complete_LLM/


└── Auto-Complete-LLM-from-the-scratch-main

    ├── README.md
    
    ├── argparsing.py
    
    ├── bigram.ipynb
    
    ├── chatbot.py
    
    ├── chatbotTweaked.py
    
    ├── data-process.py
    
    ├── gpt-v1.ipynb
    
    ├── gpt-v1FurtherTweaked.ipynb
    
    ├── gpt-v1modified.ipynb
    
    └── instructions_to_open


---

## 🧠 Project Overview

This project builds a **GPT-style causal language model** using PyTorch from scratch and wraps it with chatbot interaction logic. It covers:
- Data preprocessing (plain text files)
- Training a character-level Bigram model
- Implementing and tuning a multi-layer Transformer (GPT)
- Chatbot interface for interactive use
- Command-line and programmatic chat modes

---

## 🔍 File-by-File Breakdown

---

### 1. `bigram.ipynb`

> **Purpose**: Implements a character-level Bigram model to establish a simple probabilistic language model baseline.

#### 📘 Key Sections:
- **Data Preparation**:
  - Loads a Shakespeare dataset.
  - Encodes characters into integer tokens.
- **Bigram Model**:
  - Constructs a simple neural network where logits for the next character are directly learned from previous character embeddings.
- **Training Loop**:
  - Optimizes using cross-entropy loss.
  - Prints loss values periodically.
- **Sampling**:
  - Demonstrates sentence generation from trained Bigram.

#### 📌 Takeaway:
A lightweight intro to language modeling — helps validate training and tokenization logic before scaling to GPT.

---

### 2. `gpt-v1.ipynb`

> **Purpose**: First working version of a transformer-based GPT model.

#### 📘 Key Sections:
- **Tokenizer Setup**:
  - Converts input corpus into character-level tokens.
- **Model Definition**:
  - Implements `Head`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, and `GPTLanguageModel` from scratch.
- **Training Loop**:
  - Uses AdamW optimizer.
  - Employs autoregressive loss via next-token prediction.
- **Sampling**:
  - Allows generating text from a start prompt.

#### 📌 Takeaway:
The core transformer architecture begins here, demonstrating positional encoding, attention, and decoder-only setup.

---

### 3. `gpt-v1modified.ipynb`

> **Purpose**: Refined version of `gpt-v1.ipynb` with architectural improvements.

#### 🔧 Improvements:
- More efficient attention implementation.
- Added dropout and layer norm.
- Cleaned up sampling method for more coherent output.
- Longer training loop for better convergence.

#### 📌 Takeaway:
Moves closer to miniGPT-2. A good checkpoint to evaluate scaling impact.

---

### 4. `gpt-v1FurtherTweaked.ipynb`

> **Purpose**: Further experimentations with:
  - Sampling techniques
  - Temperature and top-k decoding
  - Embedding visualization
  - Token frequency analysis

#### 💡 Experiments:
- Tries different configurations (e.g., varying `block_size`, `vocab_size`).
- Saves and reloads trained models for reuse in chatbot.

#### 📌 Takeaway:
Final model used in chatbot. Balances speed and coherence. Pretrained weights may be extracted from here.

---

### 5. `data-process.py`

> **Purpose**: Preprocesses raw text data and prepares it for training.

#### 🔍 Functionality:
- Tokenizes input file.
- Converts text to character indices.
- Splits into training and validation sets.
- Returns data as PyTorch tensors (`x`, `y`) for use in GPT.

#### 🔧 Usage:
Imported directly by notebooks and chatbot scripts.

---

### 6. `argparsing.py`

> **Purpose**: Command-line interface parser for running the chatbot via terminal.

#### 📘 Features:
- Allows toggling between:
  - Console vs. file-based interaction
  - Temperature / top-k / max tokens
  - Logging options
- Defines command-line arguments like:
  - `--prompt`, `--temperature`, `--max_tokens`, etc.

#### ✅ Usage:
```bash
python chatbot.py --prompt "Once upon a time" --temperature 0.8
```


====================================================================

7. chatbot.py
   
====================================================================

📌 Purpose:
Command-line chatbot using the final trained GPT model.

🧠 Key Features:
- Loads the model from `.pt` or `.pth` checkpoint.
- Accepts prompt input from CLI or interactive mode.
- Generates output using sampling with temperature and top-k filtering.

💬 Example:
------------------------------------------------------------
bash
------------------------------------------------------------
User: What is your name?
Bot: I am a model of many words, yet no name to call my own...

📦 Notes:
- Uses `argparsing.py` for CLI configuration.
- Suitable for offline interactions.


====================================================================

8. chatbotTweaked.py
   
====================================================================

📌 Purpose:
Enhanced version of the chatbot with better UX and modular logic.

🆚 Differences from chatbot.py:
- Adds typing animation.
- Better memory of past context.
- Optional logging of conversations.
- Code cleanliness improved for extension (e.g., Discord bot).

🛠️ Good for:
- Plug-and-play chatbot usage
- Deploying to frontends or messengers


====================================================================

9. README.md
    
====================================================================

📌 Purpose:
Placeholder README file (now replaced with a detailed project overview).


====================================================================

10. instructions_to_open
    
====================================================================

📌 Purpose:
A simple help text for users unfamiliar with `.ipynb` files.

📎 Might say something like:

------------------------------------------------------------

mathematica

------------------------------------------------------------
Open .ipynb files in Google Colab or Jupyter Notebook



====================================================================

⚙️ Setup Instructions

====================================================================

1. Clone Repository:
------------------------------------------------------------
bash
------------------------------------------------------------
git clone https://github.com/sp2772/Auto-Complete-LLM-from-the-scratch.git
cd Auto-Complete-LLM-from-the-scratch

2. Install Dependencies:
------------------------------------------------------------
bash
------------------------------------------------------------
pip install torch numpy tqdm

3. Run Model Training (Optional):
- Use any of the `gpt-v1*.ipynb` notebooks in Jupyter/Colab to retrain models.

4. Launch Chatbot:
------------------------------------------------------------
bash
------------------------------------------------------------
python chatbotTweaked.py --prompt "Hello there" --temperature 0.9



====================================================================

🧪 Sample Outputs

====================================================================

------------------------------------------------------------
vbnet
------------------------------------------------------------
User: Tell me a story
Bot: Once upon a time, in a land ruled by silence and stars, a whisper became a song...

User: Who are you?
Bot: I am but a swirl of thoughts stitched together by patterns.



====================================================================

💬 Author

====================================================================

SP2772  
GitHub: https://github.com/sp2772
source: https://www.youtube.com/watch?v=UU1WVnMk4E8
https://www.freecodecamp.org
Course developed by ‪@elliotarledge‬ 
code sources: https://github.com/Infatoshi/fcc-intro-to-llms



====================================================================

📜 License

====================================================================

MIT License - feel free to reuse, modify, and deploy.



====================================================================

🌟 Contributions Welcome

====================================================================

If you want to:
- Add Discord integration
- Train on different datasets
- Improve attention efficiency

📮 Submit a PR or raise an issue!
"""
