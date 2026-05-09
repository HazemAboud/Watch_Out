import streamlit as st
from ultralytics import YOLO
import os
import pandas as pd
import torch
import shutil
import glob # Moved glob import here as it's used in the cache clearing section

# Page Configuration
st.set_page_config(page_title="YOLO Training Dashboard", layout="wide")

st.title("🚀 YOLOv11 Training & Evaluation Dashboard")
st.markdown("---")

st.sidebar.header("1. Dataset Configuration")
# Default path based on the location of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
default_data_path = os.path.join(current_dir, 'data', 'data.yaml')

data_path = st.sidebar.text_input("Path to data.yaml", value=default_data_path)

st.sidebar.header("2. Model & Preprocessing")
model_name = st.sidebar.selectbox("Base Model", ["yolo11s.pt", "yolo11n.pt", "yolo11m.pt", "yolo11l.pt"], index=0)
imgsz = st.sidebar.number_input("Image Size (Scaling)", min_value=320, max_value=1920, value=960, step=32)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=128, value=8)

st.sidebar.header("3. Training Hyperparameters")
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=1000, value=100)
optimizer = st.sidebar.selectbox("Optimizer", ["AdamW", "SGD", "Adam"], index=0)
lr0 = st.sidebar.number_input("Initial Learning Rate (lr0)", value=0.001, format="%.4f")

device_options = ['cpu']
if torch.cuda.is_available():
    device_options.insert(0, '0')
device = st.sidebar.selectbox("Device", device_options)

tab1, tab2 = st.tabs(["Training Control", "Evaluation Metrics"])

with tab1:
    st.subheader("Training Controls")
    
    clear_cache = st.checkbox("Clear Dataset Cache (*.cache)", value=True, help="Removes .cache files to ensure fresh data loading.")
    
    if st.button("Start Training", type="primary"):
        if not os.path.exists(data_path):
            st.error(f"Data file not found at: {data_path}")
        else:
            if clear_cache:
                with st.spinner("Clearing dataset cache..."):
                    cache_pattern = os.path.join(os.path.dirname(data_path), '**', '*.cache')
                    for cache_file in glob.glob(cache_pattern, recursive=True):
                        try:
                            os.remove(cache_file)
                        except Exception as e:
                            st.warning(f"Could not delete {cache_file}: {e}")
                st.success("Cache cleared.")

            st.info(f"Initializing {model_name} on device {device}...")
            try:
                model = YOLO(model_name)
                # Define project/name for saving
                project_name = "obstacle_tuning_gui"
                run_name = f"{model_name[:-3]}_{optimizer}_lr{lr0}_b{batch_size}"
                with st.spinner(f"Training in progress... (Epochs: {epochs})"):
                    # Create a placeholder for logs (optional, capturing stdout is complex in Streamlit, keeping it simple)
                    st.text("Training started. Please check the terminal for real-time logs.")
                    
                    results = model.train(
                        data=data_path,
                        epochs=epochs,
                        imgsz=imgsz,
                        batch=batch_size,
                        optimizer=optimizer,
                        lr0=lr0,
                        device=device,
                        project=project_name,
                        name=run_name,
                        patience=50,
                        cos_lr=True,
                        warmup_epochs=3,
                        close_mosaic=10,
                        verbose=True,
                        exist_ok=True # Overwrite existing run with same name for dashboard simplicity
                    )
                
                st.success("Training Complete!")
                save_dir = results.save_dir
                best_weight_path = os.path.join(save_dir, 'weights', 'best.pt')
                st.session_state['last_run_dir'] = save_dir
                st.session_state['best_weight_path'] = best_weight_path
                
                st.write(f"Results saved to: `{save_dir}`")
                
            except Exception as e:
                st.error(f"An error occurred during training: {e}")

with tab2:
    st.subheader("Training Visualization & Metrics")
    
    run_dir = st.session_state.get('last_run_dir')
    
    if run_dir and os.path.exists(run_dir):
        results_csv = os.path.join(run_dir, 'results.csv')
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            # Clean column names (strip spaces)
            df.columns = [c.strip() for c in df.columns]
            
            # Display Key Metrics
            st.markdown("### Final Epoch Metrics")
            last_row = df.iloc[-1]
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("mAP@50", f"{last_row.get('metrics/mAP50(B)', 0):.4f}")
            m2.metric("mAP@50-95", f"{last_row.get('metrics/mAP50-95(B)', 0):.4f}")
            m3.metric("Precision", f"{last_row.get('metrics/precision(B)', 0):.4f}")
            m4.metric("Recall", f"{last_row.get('metrics/recall(B)', 0):.4f}")
            
            st.markdown("### Training Curves")
            st.markdown("**Losses**")
            loss_cols = [c for c in df.columns if 'loss' in c]
            if loss_cols:
                st.line_chart(df[loss_cols])
            
            st.markdown("**Accuracy (mAP)**")
            map_cols = [c for c in df.columns if 'mAP' in c]
            if map_cols:
                st.line_chart(df[map_cols])
                
            # Confusion Matrix (if available)
            cm_path = os.path.join(run_dir, 'confusion_matrix.png')
            if os.path.exists(cm_path):
                st.markdown("### Confusion Matrix")
                st.image(cm_path, caption="Confusion Matrix")
                
        else:
            st.warning("results.csv not found in the run directory.")
    else:
        st.info("No training run detected yet. Go to the 'Training Control' tab to start.")

st.markdown("---")
st.caption("To run this dashboard: `streamlit run e:\\WO\\train_gui.py`")