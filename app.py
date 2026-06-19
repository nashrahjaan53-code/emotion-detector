import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import pandas as pd

# ── page configuration ────────────────────────────────
st.set_page_config(
    page_title="Emotion Sense AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom premium styling ────────────────────────────
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #0e1117;
    }
    
    /* Title and Header styling */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(45deg, #FF512F, #DD2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #8892b0;
        margin-bottom: 30px;
        font-family: 'Inter', sans-serif;
    }

    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 81, 47, 0.4);
    }
    
    /* Custom emotion bar */
    .bar-container {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .bar-fill {
        height: 12px;
        border-radius: 8px;
        transition: width 0.6s ease-in-out;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0b0c10;
    }
</style>
""", unsafe_allow_html=True)

# ── emotion color map ─────────────────────────────────
color_map = {
    'happy'   : '#00FF00',  # Green
    'sad'     : '#FF6400',  # Orange
    'angry'   : '#FF0000',  # Red
    'surprise': '#00FFFF',  # Cyan
    'neutral' : '#C8C8C8',  # Gray
    'fear'    : '#800080',  # Purple
    'disgust' : '#008000',  # Dark Green
}

emotion_emojis = {
    'happy'   : "😊 Happy",
    'sad'     : "😢 Sad",
    'angry'   : "😠 Angry",
    'surprise': "😲 Surprised",
    'neutral' : "😐 Neutral",
    'fear'    : "😨 Fearful",
    'disgust' : "🤢 Disgusted",
}

# ── cache model load ──────────────────────────────────
@st.cache_resource
def load_model():
    # Pre-loads the Emotion model to memory
    return DeepFace.build_model("Emotion")

with st.spinner("Initializing DeepFace Models... Please wait."):
    model = load_model()

# ── app layout ────────────────────────────────────────
st.markdown("<h1>🎭 Emotion Sense AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time facial expression analysis powered by Deep Learning</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=80)
st.sidebar.title("Configuration")
detector_backend = st.sidebar.selectbox(
    "Face Detector Backend",
    ["opencv", "ssd", "mtcnn", "retinaface"],
    help="Select the backend library used to locate faces in the image."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to Run Locally
1. Install requirements:
   `pip install -r requirements.txt`
2. Run streamlit:
   `streamlit run app.py`
""")

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Webcam Snapshot", "📤 Upload Image", "📖 Details & Architecture"])

# ── image processing function ─────────────────────────
def analyze_image(img_np):
    # Convert RGB (from PIL/Streamlit) to BGR (for OpenCV/DeepFace)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Run DeepFace analysis
    try:
        results = DeepFace.analyze(
            img_path=img_bgr,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend=detector_backend,
            silent=True
        )
        return results, img_bgr
    except Exception as e:
        # Fallback if face is not detected or enforcement fails
        try:
            results = DeepFace.analyze(
                img_path=img_bgr,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=detector_backend,
                silent=True
            )
            return results, img_bgr
        except Exception as ex:
            return None, img_bgr

def display_results(results, img_bgr):
    if not results:
        st.error("Could not analyze facial emotions. Please ensure your face is clearly visible and well-lit.")
        return

    result = results[0]
    emotions = result['emotion']
    dominant = result['dominant_emotion']
    region = result['region']
    
    # Draw bounding box
    color_hex = color_map.get(dominant, '#FFFFFF')
    # Hex to BGR tuple for OpenCV
    color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color_bgr, 3)
    
    # Display image & metrics side-by-side
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        # Convert back to RGB for Streamlit display
        img_annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_annotated, caption="Processed Image", use_container_width=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Primary Emotion</h3>
            <h2 style="color: {color_hex}; margin-top: 0; margin-bottom: 10px;">{emotion_emojis.get(dominant, dominant).upper()}</h2>
            <p style="font-size: 0.9rem; color: #8892b0;">Confidence: {emotions[dominant]:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Emotion Breakdown")
        
        # Sort emotions by score
        sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
        
        for emo_name, score in sorted_emotions:
            emoji_label = emotion_emojis.get(emo_name, emo_name)
            bar_color = color_map.get(emo_name, '#FFFFFF')
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 4px;">
                <span>{emoji_label}</span>
                <span style="font-weight: bold; color: {bar_color};">{score:.1f}%</span>
            </div>
            <div class="bar-container">
                <div class="bar-fill" style="width: {score}%; background-color: {bar_color};"></div>
            </div>
            """, unsafe_allow_html=True)

# ── tab 1: snapshot ───────────────────────────────────
with tab1:
    st.write("Capture a frame from your webcam to identify emotions instantly.")
    camera_image = st.camera_input("Take a photo")
    
    if camera_image is not None:
        # Load image with PIL
        img = Image.open(camera_image)
        img_np = np.array(img)
        
        with st.spinner("Analyzing image..."):
            results, img_bgr = analyze_image(img_np)
            display_results(results, img_bgr)

# ── tab 2: upload ─────────────────────────────────────
with tab2:
    st.write("Upload a portrait photo (JPG, PNG, or JPEG) to analyze.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_np = np.array(img)
        
        with st.spinner("Analyzing uploaded image..."):
            results, img_bgr = analyze_image(img_np)
            display_results(results, img_bgr)

# ── tab 3: info ───────────────────────────────────────
with tab3:
    st.markdown("""
    ### Technical Architecture
    
    This application utilizes **DeepFace**, a hybrid deep-learning facial analysis framework.
    
    1. **Face Detection Backend**: Locates the face bounding box in the input image. You can toggle between different backends in the sidebar:
       - **OpenCV (default)**: Fast and CPU-friendly, ideal for standard environments.
       - **SSD**: Balanced speed and accuracy.
       - **MTCNN**: Multi-task Cascaded Convolutional Networks, highly robust to rotation and lighting.
       - **RetinaFace**: State-of-the-art detector, extremely accurate but computationally heavier.
       
    2. **Emotion Classification**: The cropped facial region is passed to a custom convolutional neural network (CNN) trained specifically to categorize expressions into 7 facial muscle activation patterns (emotions).
    
    3. **UI Elements**:
       - Bounding boxes are dynamically drawn around detected faces.
       - The bounding box color reflects the dominant emotion.
       - The breakdown is rendered as smooth CSS bar indicators matching the exact HSL/RGB colors associated with each emotion.
    """)
