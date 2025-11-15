import io
import string

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from audio_recorder_streamlit import audio_recorder
from plotly.subplots import make_subplots
import streamlit as st
from transformers import (
    BertModel,
    BertTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


st.set_page_config(
    page_title="Scaled Dot-Product Attention Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
    }
    section.main > div {
        padding-top: 1.5rem;
    }
    .metric-card {
        border: 1px solid #1E293B;
        border-radius: 0.75rem;
        padding: 1rem;
        background-color: #1E293B;
        color: #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_NAME = "bert-large-uncased"
WHISPER_MODEL_NAME = "openai/whisper-tiny.en"
LAYER_TO_VISUALIZE = 20
HEAD_TO_VISUALIZE = 7


MANUAL_TOKENS = ["robot", "ate", "apple"]
MANUAL_X = np.array(
    [
        [0.8, 0.1, 0.3, 0.2],
        [0.1, 0.9, 0.2, 0.7],
        [0.6, 0.2, 0.8, 0.4],
    ]
)
MANUAL_WQ = np.array(
    [
        [0.6, -0.1],
        [-0.3, 0.7],
        [0.4, 0.2],
        [0.1, 0.5],
    ]
)
MANUAL_WK = np.array(
    [
        [0.2, -0.4],
        [0.7, 0.1],
        [0.3, 0.6],
        [-0.2, 0.4],
    ]
)
MANUAL_WV = np.array(
    [
        [0.4, 0.0, 0.3],
        [-0.1, 0.5, 0.2],
        [0.3, 0.1, 0.4],
        [0.2, -0.2, 0.1],
    ]
)

PITCH_SENTENCE = "the robot ate the apple because it was tasty".split()
PITCH_EMBEDDINGS = {
    "the": [0.1, 0.1, 0.1, 0.1, 0.1],
    "robot": [0.9, 0.2, 0.0, 0.0, 0.2],
    "ate": [0.1, 0.8, 0.1, 0.0, 0.1],
    "apple": [0.1, 0.1, 0.9, 0.0, 0.6],
    "because": [0.2, 0.1, 0.1, 0.1, 0.1],
    "it": [0.1, 0.1, 0.6, 0.8, 0.5],
    "was": [0.1, 0.3, 0.2, 0.0, 0.1],
    "tasty": [0.0, 0.1, 0.8, 0.0, 0.7],
}
PITCH_WQ = np.array(
    [
        [0.6, 0.1, 0.0],
        [0.0, 0.9, 0.0],
        [0.4, 0.0, 0.4],
        [0.5, 0.0, 0.6],
        [0.0, 0.5, 0.1],
    ]
)
PITCH_WK = np.array(
    [
        [0.2, 0.2, 0.0],
        [0.0, 0.7, 0.1],
        [0.7, 0.0, 0.5],
        [0.3, 0.0, 0.6],
        [0.0, 0.4, 0.1],
    ]
)
PITCH_WV = np.array(
    [
        [0.3, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.4, 0.0, 0.3],
        [0.3, 0.0, 0.1],
        [0.0, 0.3, 0.1],
    ]
)

DEFAULT_SENTENCE = "The robot ate the apple because it was tasty."

if "typed_text" not in st.session_state:
    st.session_state["typed_text"] = DEFAULT_SENTENCE
if "overflow_values" not in st.session_state:
    st.session_state["overflow_values"] = "100, 1000, 100"


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def tiny_attention(
    tokens,
    X,
    W_q,
    W_k,
    W_v,
    mask_diagonal: bool = False,
):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    dk = W_k.shape[1]
    scores = Q @ K.T / np.sqrt(dk)
    if mask_diagonal:
        scores = scores - np.eye(len(tokens)) * 1e6
    weights = softmax(scores)
    output = weights @ V
    return {
        "tokens": tokens,
        "X": X,
        "Q": Q,
        "K": K,
        "V": V,
        "scores": scores,
        "weights": weights,
        "output": output,
    }


def matrix_df(matrix, row_labels, col_labels):
    df = pd.DataFrame(np.round(matrix, 3), index=row_labels, columns=col_labels)
    return df


def render_matrix(title, matrix, row_labels, col_labels):
    st.markdown(f"**{title}**")
    df = matrix_df(matrix, row_labels, col_labels)
    st.dataframe(df, use_container_width=True)
    st.caption(f"shape = {matrix.shape[0]} x {matrix.shape[1]}")


def attention_heatmap(weights, tokens, title, height=420):
    fig = go.Figure(
        data=go.Heatmap(
            z=weights,
            x=tokens,
            y=tokens,
            colorscale="Viridis",
            hovertemplate="<b>Query:</b> %{y}<br><b>Key:</b> %{x}<br><b>Weight:</b> %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Key (column)",
        yaxis_title="Query (row)",
        yaxis_autorange="reversed",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font_color="#E2E8F0",
        height=height,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def scanning_plot(scores, tokens, query_idx, title):
    values = scores[query_idx]
    fig = go.Figure(
        data=go.Bar(
            x=tokens,
            y=values,
            marker_color="#60A5FA",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Key vector",
        yaxis_title="Raw dot product",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font_color="#E2E8F0",
        height=360,
    )
    return fig


@st.cache_resource
def load_transformer_models():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = BertModel.from_pretrained(MODEL_NAME, output_attentions=True)
    bert_model.eval()

    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME)
    whisper_model.eval()

    return tokenizer, bert_model, whisper_processor, whisper_model


def transcribe_audio_bytes(audio_bytes, whisper_processor, whisper_model):
    if not audio_bytes:
        return ""
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_data, _ = librosa.load(audio_file, sr=16000, mono=True)
        audio_data = librosa.util.normalize(audio_data)
        inputs = whisper_processor(audio_data, sampling_rate=16000, return_tensors="pt")
        decoder_prompt = whisper_processor.get_decoder_prompt_ids(language="en", task="transcribe")
        with torch.no_grad():
            predicted_ids = whisper_model.generate(
                inputs.input_features,
                forced_decoder_ids=decoder_prompt,
                max_new_tokens=128,
            )
        text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception as exc:
        st.error(f"Whisper transcription failed: {exc}")
        return ""


def get_head_details(text, tokenizer, model, layer_idx, head_idx):
    clean_text = (text or "").strip()
    if not clean_text:
        return None
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    try:
        weights_full = outputs.attentions[layer_idx][0, head_idx].detach().cpu().numpy()
    except IndexError:
        st.error(
            f"Layer/head index out of range. This model exposes layers 0-{len(outputs.attentions)-1} and heads 0-{outputs.attentions[0].shape[1]-1}."
        )
        return None

    hidden_states_all = outputs.hidden_states  # embeddings + per-layer outputs
    if layer_idx >= len(hidden_states_all):
        return None
    layer_input = hidden_states_all[layer_idx]
    batch_size, seq_len, hidden_dim = layer_input.shape
    n_heads = getattr(model.config, "n_heads", None) or getattr(model.config, "num_attention_heads")
    head_dim = hidden_dim // n_heads

    if hasattr(model, "encoder"):
        layer_module = model.encoder.layer[layer_idx]
        attention_block = layer_module.attention
        q_linear = attention_block.self.query
        k_linear = attention_block.self.key
        v_linear = attention_block.self.value
    else:
        layer_module = model.transformer.layer[layer_idx]
        attention_block = layer_module.attention
        q_linear = attention_block.q_lin
        k_linear = attention_block.k_lin
        v_linear = attention_block.v_lin

    def project_and_slice(linear_layer):
        projected = linear_layer(layer_input)
        projected = projected.view(batch_size, seq_len, n_heads, head_dim)
        return projected[0, :, head_idx, :].detach().cpu().numpy()

    q_head = project_and_slice(q_linear)
    k_head = project_and_slice(k_linear)
    v_head = project_and_slice(v_linear)

    q_weight = q_linear.weight.detach().cpu().numpy()
    k_weight = k_linear.weight.detach().cpu().numpy()
    v_weight = v_linear.weight.detach().cpu().numpy()
    head_slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
    Wq_head = q_weight[:, head_slice]
    Wk_head = k_weight[:, head_slice]
    Wv_head = v_weight[:, head_slice]

    bad_tokens = set(string.punctuation) | {"[CLS]", "[SEP]", "[PAD]", "\u00ab", "\u00bb"}
    keep_indices = [i for i, token in enumerate(token_list) if token not in bad_tokens]
    if len(keep_indices) < 2:
        keep_indices = list(range(len(token_list)))
    filtered_tokens = [token_list[i] for i in keep_indices]

    hidden_filtered = layer_input[0, keep_indices, :].detach().cpu().numpy()
    q_filtered = q_head[keep_indices]
    k_filtered = k_head[keep_indices]
    v_filtered = v_head[keep_indices]

    scores_full = q_head @ k_head.T / np.sqrt(head_dim)
    context_full = weights_full @ v_head

    filtered_scores = scores_full[np.ix_(keep_indices, keep_indices)]
    filtered_weights = weights_full[np.ix_(keep_indices, keep_indices)]
    filtered_output = context_full[keep_indices]

    return {
        "text": clean_text,
        "token_list": token_list,
        "filtered_tokens": filtered_tokens,
        "filtered_scores": filtered_scores,
        "filtered_weights": filtered_weights,
        "hidden_filtered": hidden_filtered,
        "q_vectors": q_filtered,
        "k_vectors": k_filtered,
        "v_vectors": v_filtered,
        "context_filtered": filtered_output,
        "scores_full": scores_full,
        "weights_full": weights_full,
        "Wq_head": Wq_head,
        "Wk_head": Wk_head,
        "Wv_head": Wv_head,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
    }


def render_model_attention(details, label):
    if not details:
        st.warning("Provide text to visualize attention.")
        return
    attention_heatmap(
        details["filtered_weights"],
        details["filtered_tokens"],
        label,
        height=520,
    )
    st.caption("Direct slice of the model's attention (rows may not sum to 1 after filtering).")
    with st.expander("See tokenizer output (before filtering)"):
        st.code(" | ".join(details["token_list"]))


@st.cache_data
def variance_curves(max_dim=64, samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    dims = np.arange(2, max_dim + 1, 2)
    var_raw = []
    var_scaled = []
    for d in dims:
        a = rng.standard_normal((samples, d))
        b = rng.standard_normal((samples, d))
        dots = np.sum(a * b, axis=1)
        var_raw.append(float(np.var(dots)))
        scaled = dots / np.sqrt(d)
        var_scaled.append(float(np.var(scaled)))
    return dims, np.array(var_raw), np.array(var_scaled)


@st.cache_data
def pitch_heatmap():
    X = np.array([PITCH_EMBEDDINGS[token] for token in PITCH_SENTENCE])
    result = tiny_attention(
        PITCH_SENTENCE,
        X,
        PITCH_WQ,
        PITCH_WK,
        PITCH_WV,
        mask_diagonal=True,
    )
    return result


def gradient_figure():
    x = np.linspace(-10, 10, 400)
    logits = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    probs = softmax(logits)[:, 0]
    gradient = probs * (1 - probs)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=x, y=probs, name="softmax([x,0,0])", line=dict(color="#60A5FA")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gradient,
            name="derivative",
            line=dict(color="#F472B6", dash="dot"),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Softmax confidence vs. gradient",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font_color="#E2E8F0",
        height=420,
    )
    fig.update_xaxes(title_text="logit for the highlighted token")
    fig.update_yaxes(title_text="probability", secondary_y=False)
    fig.update_yaxes(title_text="gradient", secondary_y=True)
    return fig


def overflow_demo(values):
    arr = np.array(values, dtype=float)
    raw_exp = np.exp(arr)
    shifted = arr - np.max(arr)
    safe_exp = np.exp(shifted)
    stable = safe_exp / np.sum(safe_exp)
    return raw_exp, shifted, safe_exp, stable


def parse_values(text):
    try:
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError:
        return None


def render_walkthrough(details, token_display_limit, feature_display_dim, query_token=None):
    if not details:
        st.warning("Enter text and pick a layer/head to run the NumPy walkthrough.")
        return
    tokens = details["filtered_tokens"]
    if not tokens:
        st.warning("No tokens available after filtering special characters.")
        return
    token_count = min(len(tokens), max(1, token_display_limit))
    token_subset = tokens[:token_count]
    embedding_dim = min(feature_display_dim, details["hidden_dim"])
    head_dim = min(feature_display_dim, details["head_dim"])

    def slice_tokens(matrix, cols=None):
        subset = matrix[:token_count]
        if cols:
            subset = subset[:, :cols]
        return subset

    st.markdown("### Step-by-step linear algebra (actual Transformer values)")
    render_matrix(
        "Layer input X",
        slice_tokens(details["hidden_filtered"], embedding_dim),
        token_subset,
        [f"x{i+1}" for i in range(embedding_dim)],
    )
    st.caption("Showing only the first few feature columns for readability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_matrix(
            "Wq (head slice)",
            details["Wq_head"][:embedding_dim, :head_dim],
            [f"x{i+1}" for i in range(embedding_dim)],
            [f"q{i+1}" for i in range(head_dim)],
        )
    with col2:
        render_matrix(
            "Wk (head slice)",
            details["Wk_head"][:embedding_dim, :head_dim],
            [f"x{i+1}" for i in range(embedding_dim)],
            [f"k{i+1}" for i in range(head_dim)],
        )
    with col3:
        render_matrix(
            "Wv (head slice)",
            details["Wv_head"][:embedding_dim, :head_dim],
            [f"x{i+1}" for i in range(embedding_dim)],
            [f"v{i+1}" for i in range(head_dim)],
        )

    st.markdown("#### Learned perspectives Q, K, V")
    q_col, k_col, v_col = st.columns(3)
    with q_col:
        render_matrix(
            "Q = X @ Wq",
            slice_tokens(details["q_vectors"], head_dim),
            token_subset,
            [f"q{i+1}" for i in range(head_dim)],
        )
    with k_col:
        render_matrix(
            "K = X @ Wk",
            slice_tokens(details["k_vectors"], head_dim),
            token_subset,
            [f"k{i+1}" for i in range(head_dim)],
        )
    with v_col:
        render_matrix(
            "V = X @ Wv",
            slice_tokens(details["v_vectors"], head_dim),
            token_subset,
            [f"v{i+1}" for i in range(head_dim)],
        )

    scores_subset = details["filtered_scores"][:token_count, :token_count]
    weights_subset = details["filtered_weights"][:token_count, :token_count]
    render_matrix("Score = QK^T / sqrt(d_k)", scores_subset, token_subset, token_subset)
    render_matrix("Softmax weights (actual head output)", weights_subset, token_subset, token_subset)
    attention_heatmap(weights_subset, token_subset, "Heatmap of attention weights")

    chosen_query = query_token if query_token in details["filtered_tokens"] else token_subset[0]
    if chosen_query not in token_subset:
        chosen_query = token_subset[0]
    query_idx = token_subset.index(chosen_query)
    scan_fig = scanning_plot(
        scores_subset,
        token_subset,
        query_idx,
        f"Dot products for query '{chosen_query}'",
    )
    st.plotly_chart(scan_fig, use_container_width=True)

    st.markdown("#### Context-rich output (weights @ V)")
    render_matrix(
        "Output",
        details["context_filtered"][:token_count, :head_dim],
        token_subset,
        [f"v{i+1}" for i in range(head_dim)],
    )


st.title("Attention is a Weighted Average - Live Demo")
st.caption("A 3-part walkthrough for math-focused presentations")

st.header("Interactive Transformer Attention Explorer")
st.write(
    "Record or upload speech (Whisper handles the transcript) or type text manually, then"
    " inspect a BERT attention head in real time while rerunning the linear-algebra walkthrough."
)

with st.spinner("Loading BERT and Whisper models..."):
    tokenizer, bert_model, whisper_processor, whisper_model = load_transformer_models()
st.success("Models ready. Use the fixed left panel to tweak inputs while the right panel updates immediately.")

head_details = None
selected_query = None

with st.sidebar:
    st.header("Control panel")
    st.markdown("##### BERT attention")
    num_layers = bert_model.config.num_hidden_layers
    max_layer_idx = num_layers - 1
    layer_default = min(LAYER_TO_VISUALIZE, max_layer_idx)
    num_heads_cfg = getattr(bert_model.config, "n_heads", None) or getattr(
        bert_model.config, "num_attention_heads"
    )
    if num_heads_cfg is None:
        num_heads_cfg = 1
    max_head_idx = num_heads_cfg - 1
    head_default = min(HEAD_TO_VISUALIZE, max_head_idx)
    layer_idx = st.slider("Layer index", 0, max_layer_idx, layer_default)
    head_idx = st.slider("Head index", 0, max_head_idx, head_default)

    st.markdown("##### Type or paste text")
    typed_input = st.text_area("Sentence", key="typed_text", height=160)
    typed_text_value = typed_input.strip()
    st.caption("The same text drives both BERT and the NumPy walkthrough.")

    st.markdown("##### Record or upload speech")
    audio_bytes = audio_recorder(text="Click to record", icon_size="2x")
    uploaded_audio = st.file_uploader(
        "Or upload a WAV/MP3/M4A/OGG file",
        type=["wav", "mp3", "m4a", "ogg"],
        key="audio_uploader",
    )
    uploaded_bytes = uploaded_audio.getvalue() if uploaded_audio else None
    audio_mime = uploaded_audio.type if uploaded_audio else "audio/wav"
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
    if uploaded_bytes:
        st.audio(uploaded_bytes, format=audio_mime)
    if st.button("Transcribe audio to text", key="audio_to_text"):
        selected_audio = audio_bytes or uploaded_bytes
        if not selected_audio:
            st.warning("Record or upload audio first.")
        else:
            with st.spinner("Running Whisper transcription..."):
                transcription = transcribe_audio_bytes(
                    selected_audio,
                    whisper_processor,
                    whisper_model,
                )
            if transcription:
                st.session_state["typed_text"] = transcription
                st.success("Transcribed audio and updated the textbox.")
                st.experimental_rerun()

    if typed_text_value:
        head_details = get_head_details(
            typed_text_value,
            tokenizer,
            bert_model,
            layer_idx,
            head_idx,
        )
    else:
        head_details = None

    st.markdown("##### Display controls")
    token_display_limit = st.slider("Max tokens to display", 2, 20, 8)
    feature_display_dim = st.slider("Feature columns to display", 2, 16, 6)

    if head_details and head_details["filtered_tokens"]:
        token_options = head_details["filtered_tokens"]
        default_token = st.session_state.get("query_token_value", token_options[0])
        if default_token not in token_options:
            default_token = token_options[0]
        selected_query = st.selectbox(
            "Query token for the scanning diagram",
            options=token_options,
            index=token_options.index(default_token),
            key="query_token_select",
        )
        st.session_state["query_token_value"] = selected_query
    else:
        selected_query = None
        st.info("Add text to unlock the query selector.")

    st.markdown("##### Numerical stability controls")
    overflow_values_text = st.text_input(
        "Values for softmax(x - max(x)) demo",
        key="overflow_values_input",
        value=st.session_state["overflow_values"],
    )
    st.session_state["overflow_values"] = overflow_values_text

typed_text_value = st.session_state["typed_text"].strip()
st.header("I. Introduction and Motivation")
st.markdown(
    """
    - **Problem.** Consider the sentence *"The robot ate the apple because it was tasty."* How does a machine know "it" points to "apple"?
    - **Thesis.** Attention is an elegant linear algebra trick: scaled dot products build a dynamic weighted average, no magic involved.
    - **Plan.** We'll build everything with NumPy so the math, shapes, and gradients stay in the spotlight.
    """
)

st.header("BERT attention viewer")
if head_details:
    st.markdown(f"**Current text:** {head_details['text']}")
    render_model_attention(
        head_details,
        f"BERT attention heatmap (layer {layer_idx}, head {head_idx})",
    )
else:
    st.warning("Type text on the left or transcribe audio to see the attention heatmap.")

st.header("II. Mathematical Foundations - Tiny NumPy Experiment")
st.write("Directly inspect BERT's Q, K, V computations for your text (layer inputs, weights, and outputs).")
if head_details:
    render_walkthrough(
        head_details,
        token_display_limit,
        feature_display_dim,
        query_token=selected_query,
    )
else:
    st.warning("Enter text on the left (or transcribe audio) to run the Tiny NumPy experiment.")

st.header("III. Numerical Stability - Why the Scaling Matters")
st.subheader("1. sqrt(d_k) keeps the variance of dot products under control")
dims, var_raw, var_scaled = variance_curves()
var_fig = go.Figure()
var_fig.add_trace(
    go.Scatter(
        x=dims,
        y=var_raw,
        name="Var(QK^T)",
        line=dict(color="#60A5FA"),
    )
)
var_fig.add_trace(
    go.Scatter(
        x=dims,
        y=var_scaled,
        name="Var((QK^T)/sqrt(d_k))",
        line=dict(color="#FBBF24"),
    )
)
var_fig.update_layout(
    xaxis_title="d_k (dimension)",
    yaxis_title="Empirical variance",
    plot_bgcolor="#0F172A",
    paper_bgcolor="#0F172A",
    font_color="#E2E8F0",
    height=420,
)
st.plotly_chart(var_fig, use_container_width=True)
st.caption("Without the scaling, variance grows linearly with d_k and softmax saturates.")

st.subheader("2. Gradients vanish when softmax saturates")
st.plotly_chart(gradient_figure(), use_container_width=True)
st.caption("Large logits push probabilities to 0/1, so d softmax / d logit approaches 0 and learning stalls.")

st.subheader("3. The subtract-max trick prevents overflow")
values = parse_values(st.session_state["overflow_values"])
if values is None:
    st.error("Provide comma-separated numbers in the left panel.")
else:
    raw_exp, shifted, safe_exp, stable = overflow_demo(values)
    col_a, col_b = st.columns(2)
    with col_a:
        render_matrix(
            "np.exp(x) without shifting",
            raw_exp.reshape(1, -1),
            ["exp"],
            [f"x{i+1}" for i in range(len(values))],
        )
    with col_b:
        render_matrix(
            "softmax(x - max(x))",
            stable.reshape(1, -1),
            ["prob"],
            [f"x{i+1}" for i in range(len(values))],
        )
    st.caption("np.exp(1000) = inf, but subtracting the max keeps every exponent <= 1.")

st.header("Visuals for a 3-5 minute pitch")
col1, col2 = st.columns(2)
with col1:
    pitch_result = pitch_heatmap()
    attention_heatmap(
        pitch_result["weights"],
        pitch_result["tokens"],
        "Attention heatmap (diagonal masked to highlight context)",
        height=500,
    )
    st.caption("Row 'it' peaks at column 'apple', the point you want on your slide.")
with col2:
    manual_result = tiny_attention(
        MANUAL_TOKENS,
        MANUAL_X,
        MANUAL_WQ,
        MANUAL_WK,
        MANUAL_WV,
    )
    robot_idx = MANUAL_TOKENS.index("robot")
    st.plotly_chart(
        scanning_plot(
            manual_result["scores"],
            MANUAL_TOKENS,
            robot_idx,
            "q(robot) scanning k(.)",
        ),
        use_container_width=True,
    )
    st.caption("This bar chart is the scanning diagram: dot products pop out for each comparison.")

st.info(
    "Tip: grab a screenshot of the heatmap, the scanning chart, and the gradient plot above -"
    " that's your entire visual story without needing a live Whisper demo."
)
