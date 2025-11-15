## Attention Live Demo

An interactive Streamlit walkthrough that connects a lightweight Groq Whisper transcription loop to BERT's scaled dot-product attention. Every word (typed or transcribed) reruns the visualization so you can teach the math of attention live—no TensorFlow, no heavyweight inference servers.

### Key Features
- **Groq Whisper transcription** – Live microphone recordings or audio uploads stream through `whisper-large-v3` via the Groq API; transcripts update instantly and trigger a full rerender.
- **Manual text editing** – The sidebar text area updates the BERT attention heatmap and matrix walkthrough on each keystroke.
- **True BERT internals** – Pulls the active layer/head from `bert-large-uncased`, displaying the exact Q/K/V vectors, score matrix, head weights, and context outputs after token filtering.
- **Mathematical storytelling** – Includes numerical-stability plots, overflow demos, and precomputed visuals (heatmap, scanning diagram) for a 3–5 minute slide deck.

### Requirements
- Python 3.10+
- [Groq API key](https://console.groq.com/) stored in `.env` as `GROQ_API_KEY=...`
- Recommended: virtual environment (the repo ships with `venv/` but you can create your own)

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ensure your .env contains GROQ_API_KEY
echo "GROQ_API_KEY=sk_your_key" >> .env
```

### Running the App
```bash
source venv/bin/activate
streamlit run "attention live demo.py"
```

The first load will download BERT and warm up the Groq client. If Groq credentials are missing you will see a sidebar error banner.

### Using the Demo
1. **Choose a BERT layer/head** – Sliders in the left sidebar cover all 24 layers and 16 heads from `bert-large-uncased`.
2. **Type or paste text** – Each keystroke updates the attention heatmap, the math tables, and the Tiny NumPy walkthrough (which now mirrors BERT’s true tensors).
3. **Record/upload audio** – Use the built-in recorder or drag a file (WAV/MP3/M4A/OGG). Every new clip is hashed; if it’s different from the last, Groq Whisper transcribes it automatically and reruns the app.
4. **Tune display controls** – Limit the number of tokens or feature columns shown to keep tables readable, and pick a query token to highlight in the “scanning” chart.
5. **Explore the math sections** – The “Numerical Stability” plots and overflow demo use the same controls as your talk’s visuals; the final section offers screenshot-ready heatmaps and diagrams.

### Troubleshooting
- **“GROQ_API_KEY is not set”** – Confirm `.env` exists in the project root and contains the key before launching Streamlit. The app loads `.env` via `python-dotenv` on import.
- **BERT download failures** – Ensure you have a stable internet connection the first time you run; models are cached under `~/.cache/huggingface/` afterward.
- **Large tables/log warnings** – Reduce “Max tokens to display” or “Feature columns to display” in the sidebar if you paste long passages.

### Project Structure
```
attention live demo.py   # Streamlit application
requirements.txt        # Python dependencies
README.md               # This guide
streamlit.log           # Runtime logs (ignored by git)
venv/                   # Optional virtual environment
```

### License
This demo is provided as-is for educational purposes. Replace the Groq key with your own credentials before sharing or deploying.
