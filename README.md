
# Project Report: Cyber Threat Intelligence System (CTI)

**Author:** Bhuvanesh
**Project:** Final Year Engineering Project
**Date:** November 5, 2025

---

## 1. Executive Summary

The Cyber Threat Intelligence System (CTI) is a comprehensive, multi-modal security platform designed to detect a wide range of digital threats in real-time. The system integrates four distinct, specialized machine learning models to provide a holistic defense against modern cyber threats, including deepfakes, audio synthesis, email phishing, and malicious URLs.

The project consists of two main components: a central web application and a browser extension. The web application serves as a dashboard for analysis, user management, and viewing historical data. The browser extension brings the system's detection capabilities directly into the user's workflow, primarily within email clients like Gmail and Outlook, enabling both manual and automatic scanning of content.

The CTI system leverages a sophisticated architecture, combining deep learning models (EfficientNet, BERT) for complex pattern recognition in media and text, and traditional machine learning models (Random Forest) for robust classification of structured feature data. Furthermore, the system enhances user understanding by integrating Google's Gemini Pro LLM to provide clear, concise, human-readable explanations for the verdicts delivered by the local models. This project demonstrates a practical and powerful application of integrated AI for advanced cybersecurity.

---

## 2. System Architecture

The CTI platform is built on a client-server model, with a modular backend that allows for independent development and scaling of its core components.

### 2.1. High-Level Diagram

```
[User] <--> [Browser Extension] <--> [Web Dashboard (Flask)] <--> [Analysis Engines]
   |                                      ^
   |                                      |
   +--------------------------------------+
```

### 2.2. Components

*   **Web Application (Backend):** A central Flask server (`app.py`) that acts as the brain of the operation. It handles:
    *   HTTP requests from the web dashboard and browser extension.
    *   User authentication and session management.
    *   Orchestration of the four analysis engines.
    *   Communication with the Gemini API for generating explanations.
    *   Serving the frontend dashboard.

*   **Frontend:**
    *   **Web Dashboard:** A user interface built with HTML, CSS, and JavaScript, served by Flask. It allows users to upload files, submit URLs, and view detailed analysis reports and history.
    *   **Browser Extension (`cti-url-scanner-extension`):** A Chromium-based extension that provides in-browser threat detection. It communicates with the Flask backend via a secure, token-authenticated API.

*   **AI/ML Analysis Engines:** Four independent Python modules, each responsible for a specific type of threat detection:
    1.  `deepfake_video_engine`
    2.  `deepfake_audio_engine`
    3.  `email_engine`
    4.  `url_engine`

### 2.3. Data Flow

1.  **Input:** A user provides input either through the web dashboard (uploading a file) or the browser extension (opening an email, clicking a scan button).
2.  **Request:** The frontend sends the data (file, text, or URL) to the Flask backend. The browser extension includes a JWT authentication token with its requests.
3.  **Orchestration:** The Flask `app.py` receives the request, identifies the required analysis type, and calls the appropriate engine (e.g., `analyze_video` for a video file).
4.  **Analysis:** The engine processes the data, loads its specific model, and returns a structured result (verdict, confidence score).
5.  **Explanation (Gemini):** The backend sends the model's verdict and data to the Gemini API to generate a qualitative explanation.
6.  **Response:** The backend combines the model's verdict and the Gemini explanation into a final report. For long analyses like video processing, this is streamed back to the client in real-time using Server-Sent Events (SSE).
7.  **Presentation:** The frontend UI (dashboard or extension popup) receives the final report and displays it to the user in a formatted, easy-to-understand manner.

---

## 3. Core AI/ML Modules

### 3.1. Module 1: Deepfake Video Detection

*   **Location:** `aFull_project/deepfake_video_bhuvanesh/`
*   **Engine File:** `deepfake_video_engine.py`

**Objective:** To accurately identify videos that have been manipulated using deepfake technology, analyzing both visual and auditory components.

**Methodology:**
This module employs a sophisticated, two-pronged approach.

1.  **Visual Analysis:**
    *   **Frame Sampling:** To ensure efficiency, the engine does not analyze every frame. It intelligently samples up to 500 frames from the video. For short videos, all frames are used; for longer videos, a random subset is selected.
    *   **Face Detection:** Each sampled frame is first passed through a ResNet-10 Single Shot Detector (SSD) face detector. This quickly identifies if a face is present, ensuring the deepfake model only processes relevant frames.
    *   **Deepfake Classification:** Frames containing a face are then passed to the core deepfake detection model, a fine-tuned **EfficientNet-B0**. This model outputs a "fake" confidence score for each frame.
    *   **Verdict Logic:** A simple average is not used. A verdict is reached based on heuristics, primarily by checking if the percentage of "suspiciously high-scored" frames exceeds a 30% threshold. This makes the detection robust against occasional misclassifications.

2.  **Audio Analysis:**
    *   The audio track is extracted from the video using `ffmpeg`.
    *   This audio is then passed to the **Deepfake Audio Detection** module for an independent analysis. The final report presents both the video and audio verdicts.

**Model Details:**
*   **Primary Model:** `EfficientNet-B0` (loaded from `ULTIMATE_CHAMPION_model.pth`). This is a powerful and efficient convolutional neural network (CNN) architecture. The model was created using transfer learning, where the final layers of a pre-trained EfficientNet were replaced and fine-tuned on a specific deepfake dataset.
*   **Face Detector:** ResNet-10 SSD Caffe model.
*   **Frameworks:** PyTorch, OpenCV DNN.

### 3.2. Module 2: Deepfake Audio Detection

*   **Location:** `aFull_project/deepfake_audio_model_rangnath/`
*   **Engine File:** `deepfake_audio_engine.py`

**Objective:** To distinguish between genuine human speech and synthesized or voice-cloned audio.

**Methodology:**
This model uses a traditional machine learning approach based on acoustic features.

1.  **Feature Extraction:** The engine loads an audio file and uses the `librosa` library to compute its **Mel-Frequency Cepstral Coefficients (MFCCs)**. MFCCs are features that represent the timbral and spectral characteristics of a voice.
2.  **Feature Aggregation:** To create a single, fixed-size input vector for the model, the *mean* of the 13 MFCCs is calculated across the entire audio clip. This provides an "average" acoustic signature of the voice.
3.  **Scaling & Prediction:** This feature vector is normalized using a pre-fitted `StandardScaler` and then fed into the trained classification model to get a prediction (FAKE/REAL) and a confidence score.

**Model Creation & Details:**
*   **Model:** The model is a **Random Forest Classifier**, a powerful ensemble learning method, loaded from `deepfake_audio_model.pkl`.
*   **Training:** The model was trained on a dataset of real and fake audio clips. For each clip, the 13 mean MFCCs were extracted. The `StandardScaler` and `RandomForestClassifier` were then trained on these features.
*   **Frameworks:** Scikit-learn, Librosa, Pickle.

### 3.3. Module 3: Email Phishing Detection

*   **Location:** `aFull_project/email_phising_tejaswi/`
*   **Engine File:** `email_engine.py`

**Objective:** To analyze the textual content of an email and determine if it is a phishing attempt.

**Methodology:**
This module leverages a state-of-the-art Natural Language Processing (NLP) model to understand the context and semantics of the email text, rather than relying on simple keyword matching.

1.  **Tokenization:** The raw email text is processed by a **BERT Tokenizer**. This converts words and sentences into a numerical format that the model can understand, padding or truncating the input to a fixed length of 128 tokens.
2.  **Inference:** The tokenized input is fed into the fine-tuned BERT model. The model processes the text, considering the relationships between words, and outputs a final classification.

**Model Creation & Details:**
*   **Model:** `TFBertForSequenceClassification` (loaded from the `saved_model` directory). This is the TensorFlow version of the powerful BERT model, designed for classification tasks.
*   **Training:** The base `bert-base-uncased` model was **fine-tuned for 3 epochs** on a specialized dataset of phishing and legitimate emails (`train.csv`). This tuning process adapts the model's general language understanding to the specific nuances of phishing emails.
*   **Frameworks:** TensorFlow, Hugging Face Transformers.
*   **Self-Sufficiency:** The project includes a local copy of the BERT tokenizer (`local_bert_tokenizer`), making it independent of an internet connection for its operation.

### 3.4. Module 4: Malicious URL Detection

*   **Location:** `aFull_project/End-to-End-Malicious-URL-Detection_NReshwar/`
*   **Engine File:** `url_engine.py`

**Objective:** To classify a given URL as either malicious or benign based on its lexical and structural features.

**Methodology:**
This model does not access the URL's content but instead performs a detailed analysis of the URL string itself.

1.  **Feature Extraction:** A custom `FeatureExtractor` class deconstructs the URL into its components (domain, path, parameters) and calculates over 80 distinct features. These include:
    *   Length of the URL, domain, path, etc.
    *   Counts of special characters (`.`, `-`, `_`, `@`, `%`, etc.).
    *   Presence of sensitive keywords (e.g., 'login', 'secure').
    *   Boolean flags (e.g., `email_in_url`, `ip_in_url`).
2.  **Prediction:** This large feature vector is then passed to a trained Random Forest model, which classifies the URL and provides a percentage-based risk score.
3.  **Whitelisting:** The system includes a whitelist of known-good domains (e.g., `google.com`, `microsoft.com`) to prevent false positives and improve performance.

**Model Creation & Details:**
*   **Model:** **Random Forest Classifier** (loaded from `random_forest.pkl`). This model is well-suited for this task as it can handle a large number of features and is robust against irrelevant ones.
*   **Training:** The model was trained on a large dataset of labeled URLs, where the lexical features were first extracted for each URL.
*   **Frameworks:** Scikit-learn, Joblib.

---

## 4. Browser Extension (`cti-url-scanner-extension`)

The browser extension is a critical component that integrates the CTI system directly into the user's daily web browsing, with a focus on email security.

### 4.1. Installation

1.  Open Google Chrome or Microsoft Edge and navigate to the extensions page (`chrome://extensions` or `edge://extensions`).
2.  Enable the **"Developer mode"** toggle, usually found in the top-right corner.
3.  Click the **"Load unpacked"** button.
4.  In the file selection dialog, navigate to and select the `D:\PROJECT\CTI\aFull_project\cti-url-scanner-extension` folder.
5.  The extension will be installed and ready to use.

### 4.2. Functionality

*   **User Authentication:** The extension has a login UI that communicates with the backend to obtain a JSON Web Token (JWT), which is securely stored and used for all subsequent API requests.
*   **Automatic Email Scanning:** This is the extension's flagship feature. Using an `email_observer.js` content script, it employs a `MutationObserver` to automatically detect when a new email is opened in Gmail or Outlook. It then transparently sends the email's content to the backend for analysis.
*   **Manual Scanning:** The user can click the "Scan Current Email" button in the extension popup to trigger an on-demand scan of the currently viewed email.
*   **Real-time User Feedback:**
    *   **Notifications:** After a scan, a system notification appears, summarizing the verdict (e.g., "🚨 Warning: 2 Malicious Link(s) Found!").
    *   **Link Decoration:** The extension injects small icons (🚨 for malicious, ✅ for safe) directly next to links within the scanned email body, providing immediate visual feedback.
*   **Live Deepfake Analysis:** The extension can use the `chrome.tabCapture` API to record the video and audio from the current tab for a set duration (e.g., 30 seconds). This recording is then streamed to the backend for deepfake analysis, allowing users to analyze content from social media, video conferencing, or any other web source.
*   **Results Popup:** The extension's main popup serves as a mini-dashboard, showing the results of the latest email and video scans.

---

## 5. Conclusion

The Cyber Threat Intelligence System successfully integrates multiple, diverse AI models into a single, cohesive platform to provide robust, real-time protection against a variety of modern digital threats. The project's modular architecture, combining a central Flask server with a powerful browser extension, creates a practical and user-friendly security tool.

By employing both deep learning and traditional machine learning techniques, the system is optimized for different tasks—using complex models like BERT and EfficientNet for nuanced content analysis and efficient models like Random Forest for structured feature classification. The addition of the Gemini API for generating explanations makes the system's outputs more transparent and valuable to the end-user.

This project serves as a strong proof-of-concept for a next-generation, AI-powered threat intelligence platform.
