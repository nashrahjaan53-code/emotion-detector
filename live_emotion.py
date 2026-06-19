import cv2
from deepface import DeepFace
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── settings ──────────────────────────────────────────
BUFFER_SIZE   = 15      # more frames = more stable
ANALYZE_EVERY = 3       # analyze more frequently
MIN_CONFIDENCE = 60     # only show if model is confident

color_map = {
    'happy'   : (0, 255, 0),
    'sad'     : (255, 100, 0),
    'angry'   : (0, 0, 255),
    'surprise': (0, 255, 255),
    'neutral' : (200, 200, 200),
    'fear'    : (128, 0, 128),
    'disgust' : (0, 128, 0),
}

emotion_buffer = []
stable_emotion = 'neutral'
stable_color   = (200, 200, 200)
stable_score   = 0.0
face_box       = (0, 0, 0, 0)
all_scores     = {}
frame_count    = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── brightness fix ────────────────────────────────────
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # boost brightness

print('Camera ON — Press Q to quit')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── auto brightness correction ────────────────────
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl    = clahe.apply(l)
    frame_bright = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    if frame_count % ANALYZE_EVERY == 0:
        try:
            res = DeepFace.analyze(
                frame_bright,           # use brightness-fixed frame
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'  # faster & more reliable
            )

            emo        = res[0]['dominant_emotion']
            all_scores = res[0]['emotion']
            top_score  = all_scores[emo]
            r          = res[0]['region']
            face_box   = (r['x'], r['y'], r['w'], r['h'])

            # only add to buffer if confidence is high enough
            if top_score >= MIN_CONFIDENCE:
                emotion_buffer.append(emo)
            else:
                emotion_buffer.append('neutral')  # low confidence = neutral

            if len(emotion_buffer) > BUFFER_SIZE:
                emotion_buffer.pop(0)

            stable_emotion = Counter(emotion_buffer).most_common(1)[0][0]
            stable_color   = color_map.get(stable_emotion, (255,255,255))
            stable_score   = top_score

        except:
            pass

    x, y, w, h = face_box
    if w > 0:
        cv2.rectangle(frame, (x, y), (x+w, y+h), stable_color, 3)
        cv2.putText(frame,
                    f'{stable_emotion.upper()}',
                    (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, stable_color, 2)

    # ── show ALL emotion scores on left side ──────────
    if all_scores:
        y_off = 30
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for emo_name, score in sorted_scores:
            bar      = '█' * int(score // 8)
            is_top   = (emo_name == stable_emotion)
            txt_col  = color_map.get(emo_name, (255,255,255)) if is_top else (180,180,180)
            prefix   = '► ' if is_top else '  '
            cv2.putText(frame,
                        f'{prefix}{emo_name:<10} {score:.0f}%',
                        (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, txt_col, 1)
            y_off += 26

    # ── confidence warning ────────────────────────────
    if stable_score < MIN_CONFIDENCE and stable_score > 0:
        cv2.putText(frame,
                    'LOW CONFIDENCE - try better lighting!',
                    (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 165, 255), 1)

    filled = len(emotion_buffer)
    bar_w  = int((filled / BUFFER_SIZE) * 200)
    cv2.rectangle(frame, (10, 455), (210, 470), (50,50,50), -1)
    cv2.rectangle(frame, (10, 455), (10 + bar_w, 470), (0,255,0), -1)
    cv2.putText(frame, f'Buffer: {filled}/{BUFFER_SIZE}',
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.putText(frame, 'Q = Quit',
                (560, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)

    cv2.imshow('Emotion Detector  |  Q = Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Done!')