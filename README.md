# Critiscan's Integrated Item Counting and Brand Detection: FMCGDetectron

## *Overview*
![image](https://github.com/user-attachments/assets/ea69c5b0-2475-436a-ba11-6ca7f3212ebc)

This repository is part of **Flipkart's Robotics Challenge Project** and focuses on **item counting** and **brand detection** within the FMCG (Fast-Moving Consumer Goods) sector. It provides robust detection capabilities by combining multiple models to process video streams and image data in real time.

- [Link to Website Repository](https://github.com/aanushkaguptaa/critiscan)
- [Link to OCR Detection Model](https://github.com/tsu-ki/ocr-script-freshness-model)
- [Link to Fruit Quality Assessment Model](https://github.com/tsu-ki/Freshness-model)

FMCGDetectron integrates seamlessly with the **OCR Detection Model**, enabling a complete system for quality assurance and inventory management. While the OCR model focuses on extracting textual information, FMCGDetectron specializes in:
1. Counting items in a frame.
2. Detecting the brand of items displayed.

---

## *Features*

- **Real-time item detection** using **YOLOv8**.
- **Brand classification** using a fine-tuned **Keras model**.
- Instance segmentation powered by **Detectron2**.
- Date detection for expiry extraction using custom patterns and NLP tools.
- Integration-ready Flask API with endpoints for camera control, video streaming, and summary retrieval.
- Multi-model architecture optimized for performance and scalability.

---

## *Architecture*

### **1. Model Components**

1. **YOLOv8**:
    - Used for item detection and counting.
    - Pre-trained and fine-tuned for FMCG-specific items.
2. **Keras Brand Classification Model**:
    - Classifies item brands from input images.
    - Utilizes a lightweight architecture with transfer learning for high accuracy.
3. **Detectron2**:
    - Performs instance segmentation to complement object detection.
    - Used for identifying masks and fine-grained details.
4. **Qwen2-VL** (Optional NLP-based component):
    - Used for conditional generation and auxiliary tasks like understanding textual prompts or captions.
5. **Custom Expiry Detection Model**:
    - Extracts expiry dates using Keras-based classification and regex patterns.

### **2. Project Workflow**

1. **Input**: Real-time video or image data from a connected camera.
2. **Preprocessing**: Frames are resized for efficient processing.
3. **Detection Pipeline**:
    - YOLOv8 detects and counts items.
    - Detectron2 segments objects to improve classification accuracy.
    - Keras Brand Model identifies the brand of items.
4. **Postprocessing**:
    - Extracts additional metadata (e.g., expiry dates).
    - Combines results for reporting.
5. **Output**:
    - Real-time overlay of detected objects and counts on the video stream.
    - JSON summaries and Excel-based logs for reporting.

---

## *API Endpoints*

The Flask-based API exposes the following endpoints:

### **1. Health Check**

- **Endpoint**: `/`
- **Description**: Confirms the server is running.

### **2. Start and Stop Camera**

- **Endpoints**:
    - `POST /start_camera`
    - `POST /stop_camera`
- **Description**: Starts or stops the video stream.

### **3. Video Feed**

- **Endpoint**: `/video_feed`
- **Description**: Streams processed frames with detections over HTTP.

### **4. Object Summary**

- **Endpoint**: `/object_summary`
- **Description**: Returns a JSON summary of detected objects and their counts.

### **5. Test API**

- **Endpoint**: `/test_api`
- **Description**: Validates model loading and camera functionality.

---

## *Installation*

### **Prerequisites**

- Python 3.8 or higher
- GPU-enabled system with CUDA (optional for faster inference)
- Libraries and frameworks:
    - TensorFlow
    - PyTorch
    - Detectron2
    - Flask
    - OpenCV
    - Transformers

### **Steps**

1. Clone the repository:
    
    ```
    git clone https://github.com/<your-repo>/fmcgdetectron.git
    cd fmcgdetectron
    ```
    
2. Install dependencies:
    
    ```
    pip install -r requirements.txt
    ```
    
3. Ensure required models are downloaded and placed in the `models/` directory:
    
    - YOLOv8 weights (`Item_count_yolov8.pt`)
    - Brand classification model (`my_model.keras`)
    - Expiry detection model (`date_detection_model1.keras`)
4. Start the server:
    
    ```
    python detect.py
    ```
    

---

## *Usage*

### **Running the System**

1. Start the Flask server:
    
    ```
    python detect.py
    ```
    
2. Open a web browser and navigate to the following URL:
    
    ```
    http://<server-ip>:5005
    ```
    

### **Processing a Video Stream**

- The system processes frames in real time, overlaying detected items, brands, and counts.
- Results are logged in an Excel file (`outputs/results.xlsx`) for analysis.

---

## *Technical Details*

### **Detection Flow**

1. **Frame Capture**:
    - Frames are captured from a live video stream or camera feed.
2. **Item Detection**:
    - **YOLOv8** identifies bounding boxes for objects.
    - Object counts are updated in real time.
3. **Brand Detection**:
    - Every 5th frame is classified using the Keras Brand Model.
    - Results are cached to optimize inference time.
4. **Instance Segmentation**:
    - Detectron2 refines detections and adds masks for more precise results.
5. **Data Logging**:
    - Detected information (timestamps, counts, brands) is saved in a structured format.

### **Performance Optimization**

- **Frame Skipping**: Processes every nth frame to reduce computational overhead.
- **Dynamic Resolution**: Reduces frame size for quicker inference.
- **GPU Acceleration**: Uses CUDA for faster processing where available.

---

## *Limitations and Future Work*

- The current pipeline assumes a controlled environment (e.g., good lighting and camera angles).
- Plans to integrate deeper brand recognition with visual-linguistic models like Qwen2-VL.
- Support for additional item categories is under consideration.
