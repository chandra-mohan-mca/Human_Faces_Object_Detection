import streamlit as st
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np
import os
import cv2
import time
import tempfile

MODEL_PATH = r"H:\GUVI_FILES\Human_Faces_Object_Detection\Faces_YOLO_Model\detect\face_yolo_exp\weights\best.pt"
model = YOLO(MODEL_PATH)

st.sidebar.title("Project Menu")
menu = st.sidebar.radio("Select", ["Data", "EDA - Visual", "Prediction", "Webcam Face Detection", "Live Camera (CCTV Mode)", "Video Prediction"])

face_count=0

if menu == "Data":
    st.header("Dataset Overview")
    # show simple dataset table
    def list_images(path):
        imgs = []
        for split in ['train','val']:
            d = os.path.join(path,'images',split)
            if os.path.exists(d):
                imgs += [(split, f) for f in os.listdir(d) if f.lower().endswith(('.jpg','.png'))]
        return imgs
    imgs = list_images(r"H:\GUVI_FILES\Human_Faces_Object_Detection\clean_dataset")
    if st.button(label="Total Images",key="total_image"):
        st.write("Total images:", len(imgs))
    if st.checkbox("Show sample images"):
        sample = imgs[:10]
        cols = st.columns(3)
        for i,(split, fn) in enumerate(sample):
            cols[i%3].image(os.path.join(r"H:\GUVI_FILES\Human_Faces_Object_Detection\clean_dataset\images",split,fn), caption=f"{split}/{fn}", use_container_width=True)

    # show model metrics CSV if you saved them
    
    if st.button(label="Check Evaluvate Matrics",key="matrics"):
        metrics_csv = 'matrics.csv'
        if os.path.exists(metrics_csv):
            df = pd.read_csv(metrics_csv)
            st.dataframe(df)
            st.snow()

elif menu == "EDA - Visual":
    st.header("EDA")
    if st.button(label="Faces Per Images and Resolution",key="faces"):
    # Example: faces per image histogram (create a small df beforehand)
        df = pd.read_csv('faces.csv')  # prepare this during preprocessing
        fl=df[:100]
        fig = px.histogram(fl, x='image_name', nbins=10, title="Faces per image")
        st.plotly_chart(fig, use_container_width=True)

        # Show resolution distribution
        df['resolution'] = df['width'] * df['height']
        ff=df[:100]
        fig2 = px.box(ff, y='resolution', title='Image resolution distribution')
        st.plotly_chart(fig2, use_container_width=True)
        st.balloons()

elif menu == "Prediction":
    st.header("Run Prediction")
    uploaded = st.file_uploader("Upload image", type=['jpg','png','jpeg'])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input", use_container_width=True)

        if st.button(label="Detect Face",key="face_detection"):
            
            with st.spinner("Please Wait..."):
                time.sleep(6)

                results = model.predict(source=np.array(img), conf=0.10, save=False)
                out = results[0].plot()
                st.image(out, caption="Detections", use_container_width=True)
        
        # show detections table
            rows=[]
            for b in results[0].boxes:
                cls = int(b.cls.cpu().numpy()[0])
                confv = float(b.conf.cpu().numpy()[0])
                x1,y1,x2,y2 = b.xyxy.cpu().numpy().tolist()[0]
                rows.append({"class": model.names[cls], "conf":round(confv,3), "x1":x1,"y1":y1,"x2":x2,"y2":y2})

                face_count += 1


        
            if rows:
                st.success(f"Number of faces detected: {face_count}")
                st.table(pd.DataFrame(rows))
                st.snow()
            else:
                st.error("No objects found.")


elif menu == "Webcam Face Detection":
    st.header("📷 Real-time Face Detection from Webcam")

    # Streamlit's built-in camera input
    camera_image = st.camera_input("Capture a photo")

    if camera_image is not None:
        img = Image.open(camera_image).convert("RGB")
        st.image(img, caption="Captured Image", use_container_width=True)

        if st.button(label="Detect Face",key="face_detection"):
            
            with st.spinner("Please Wait..."):
                time.sleep(6)


        # Run YOLO face detection
                results = model.predict(source=np.array(img), conf=0.5, save=False)
                out = results[0].plot()
                st.image(out, caption="Detected Faces", use_container_width=True)

        # Show detection table
            rows = []
            for b in results[0].boxes:
                cls = int(b.cls.cpu().numpy()[0])
                confv = float(b.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = b.xyxy.cpu().numpy().tolist()[0]
                rows.append({
                    "class": model.names[cls],
                    "conf": round(confv, 3),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

                face_count +=1

            if rows:
                st.success(f"Number of faces detected: {face_count}")
                st.table(pd.DataFrame(rows))
                st.snow()
            else:
                st.error("No objects found.")
            


elif menu == "Live Camera (CCTV Mode)":
    st.header("🎥 Live Face Detection (CCTV Mode)")
    face_counts=0
    run = st.checkbox("Start Camera")
    COUNT_PLACEHOLDER = st.empty()
    FRAME_WINDOW = st.image([])  # create a live frame display placeholder

    if run:
        cap = cv2.VideoCapture(0)  # 0 for default webcam
        st.write("✅ Camera is running... press Stop to end")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read from camera")
                break

            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            

            # Run YOLO prediction
            results = model.predict(source=frame_rgb, conf=0.3, save=False, verbose=False)


            current_faces = len(results[0].boxes)  # faces in current frame

            

            # Draw detections
            annotated_frame = results[0].plot()

            # Count faces in current frame
            face_counts = len(results[0].boxes)

            # Display
            FRAME_WINDOW.image(annotated_frame, channels="BGR")
            COUNT_PLACEHOLDER.success(f"Number of faces detected: {face_counts}")
            

            # Streamlit auto-refresh
            if not st.session_state.get("run_camera", True):
                break
        
        
        cap.release()
        st.success("🛑 Camera stopped")
        

elif menu == "Video Prediction":
    st.header("🎬 Face Detection from Video File")

    uploaded_video = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_video is not None:
        # Save uploaded file temporarily
        temp_video_path = "temp_uploaded_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.video(temp_video_path)

        st.write("⚙️ Processing video... Please wait ⏳")


        # Output file path
        output_path = "output_faces_detected.mp4"
        cap = cv2.VideoCapture(temp_video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        # Process each frame
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO inference (adjust conf for speed/accuracy)
            results = model.predict(source=frame_rgb, conf=0.10, imgsz=640, verbose=False)
            annotated = results[0].plot()

            # Convert back to BGR for saving
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            out.write(annotated_bgr)

            # Update progress
            progress.progress((i + 1) / frame_count)

        cap.release()
        out.release()

        st.success("✅ Video processing complete!")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("📥 Download Processed Video", f, file_name="faces_detected.mp4")

