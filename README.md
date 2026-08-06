# Traffic Monitoring Computer Vision

A real-time computer vision dashboard designed to monitor, classify, and track urban transit flow. Leveraging a custom-trained YOLO object detection model, this application performs multi-class vehicle detection specifically applied to Nairobi's traffic conditions (Kileleshwa). 

This project serves as a practical application of computer vision for urban traffic analysis, moving beyond static datasets to process real-world, dynamic video feeds.

##  Features

* **Real-Time Video Inference:** Processes local video feeds (`kileleshwa.mp4`) frame-by-frame using OpenCV, applying YOLO bounding boxes and generating live vehicle counts.
* **Instant Image Analysis:** Features a zero-friction UI allowing users to test the model using one-click sample images fetched via web requests, alongside standard local file uploading.
* **Custom 21-Class Detection:** Fine-tuned on the Dhaka-AI dataset to recognize a wide variety of urban transport methods, including:
  * Cars, SUVs, Minivans, and Taxis
  * Motorbikes and Scooters
  * Buses, Minibuses, and Trucks
  * Ambulances, Police Cars, and Army Vehicles
  * Auto Rickshaws, Bicycles, and Wheelbarrows
* **Dynamic Metrics Dashboard:** Automatically tallies and displays the count of each detected vehicle class in real-time using Streamlit metric badges.

##  Technology Stack

* **Core Vision Model:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (PyTorch)
* **Frontend / Dashboard:** [Streamlit](https://streamlit.io/)
* **Image / Video Processing:** OpenCV (`cv2`), Pillow (PIL), NumPy
* **Networking:** Requests (for handling web-based sample images)

##  Repository Structure

```text
├── app.py                            # Main Streamlit application script
├── yolo model/
│   └── best.pt                       # Custom-trained YOLO model weights
├── traffic feed video/
│   └── kileleshwa.mp4                # Source video for real-time inference
├── Notebooks/
│   └── Explore_traffic_flow_Data.ipynb # Model training & data exploration
├── requirements.txt                  # Python dependencies
└── README.md
