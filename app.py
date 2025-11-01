import streamlit as st
import cv2
from PIL import Image as PILImage
import concurrent.futures
import os
import time

# Import functions from existing files
import importlib.util

# Import object-detection.py (handle hyphenated filename)
spec = importlib.util.spec_from_file_location("object_detection", "object-detection.py")
object_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(object_detection)
extract_segmentation_masks = object_detection.extract_segmentation_masks

# Page configuration
st.set_page_config(
    page_title="Waste Classification & Segmentation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner UI
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
    .result-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>♻️ Waste Classification & Segmentation</h1><p>Real-time AI-powered waste analysis</p></div>', unsafe_allow_html=True)

# Initialize session state
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'gemini_response' not in st.session_state:
    st.session_state.gemini_response = None
if 'segmentation_results' not in st.session_state:
    st.session_state.segmentation_results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'live_feed_active' not in st.session_state:
    st.session_state.live_feed_active = False
if 'stop_feed' not in st.session_state:
    st.session_state.stop_feed = False
if 'capture_triggered' not in st.session_state:
    st.session_state.capture_triggered = False

# Sidebar for camera settings
with st.sidebar:
    st.header("⚙️ Settings")
    camera_index = st.selectbox("📷 Camera Index", [0, 1, 2], index=1)
    output_dir = st.text_input("📁 Output Directory", value="segmentation_outputs")
    st.markdown("---")
    
    # Status indicators
    st.header("📊 Status")
    if st.session_state.live_feed_active:
        st.success("🟢 Live Feed Active")
    else:
        st.info("⚪ Feed Inactive")
    
    if st.session_state.processing:
        st.warning("🔄 Processing...")
    elif st.session_state.gemini_response:
        st.success("✅ Analysis Complete")
    
    st.markdown("---")
    if st.button("🗑️ Clear All Results", width='stretch'):
        st.session_state.gemini_response = None
        st.session_state.segmentation_results = None
        st.session_state.captured_image = None
        st.session_state.live_feed_active = False
        st.session_state.stop_feed = True
        st.session_state.capture_triggered = False
        st.rerun()

# Main content area
col1, col2 = st.columns([1.2, 1])

with col1:
    st.header("📹 Live Camera Feed")
    
    # Camera controls
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    
    with button_col1:
        start_feed = st.button("▶️ Start Live Feed", width='stretch', type="primary")
        if start_feed:
            st.session_state.live_feed_active = True
            st.session_state.stop_feed = False
            st.session_state.capture_triggered = False
    
    with button_col2:
        stop_feed = st.button("⏸️ Stop Feed", width='stretch')
        if stop_feed:
            st.session_state.live_feed_active = False
            st.session_state.stop_feed = True
    
    with button_col3:
        capture_button = st.button("📷 Capture", width='stretch', 
                                  disabled=not st.session_state.live_feed_active)
        if capture_button and st.session_state.live_feed_active:
            st.session_state.capture_triggered = True
            st.session_state.live_feed_active = False
            st.session_state.stop_feed = True
    
    # Live feed placeholder
    live_feed_placeholder = st.empty()
    
    # Display live feed using auto-refresh
    if st.session_state.live_feed_active and not st.session_state.stop_feed:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            with live_feed_placeholder.container():
                st.error(f"❌ Cannot open camera {camera_index}. Try a different camera index.")
                st.session_state.live_feed_active = False
        else:
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                live_feed_placeholder.image(frame_rgb, caption="Live Camera Feed", channels="RGB")
                
                # Auto-refresh to show live feed
                time.sleep(0.05)  # Small delay
                st.rerun()
            else:
                with live_feed_placeholder.container():
                    st.error("❌ Failed to read from camera")
                st.session_state.live_feed_active = False
            cap.release()
    elif st.session_state.capture_triggered and not st.session_state.captured_image:
        # Capture frame when button is pressed
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                temp_image_path = "temp_captured_image.jpg"
                cv2.imwrite(temp_image_path, frame)
                st.session_state.captured_image = {
                    'path': temp_image_path,
                    'image': frame_rgb
                }
                st.session_state.capture_triggered = False
                st.success("✅ Image captured successfully!")
            cap.release()
        cv2.destroyAllWindows()
    else:
        with live_feed_placeholder.container():
            if not st.session_state.live_feed_active:
                st.info("👆 Click 'Start Live Feed' to begin")
    
    # Show captured image if available
    if st.session_state.captured_image:
        st.markdown("---")
        st.subheader("📸 Captured Image")
        st.image(st.session_state.captured_image['image'], caption="Ready to Process", channels="RGB")
        
        # Process button
        if st.button("🚀 Analyze Image", width='stretch', type="primary", 
                    disabled=st.session_state.processing):
            st.session_state.processing = True
            st.session_state.capture_triggered = False
            
            # Capture image path before threading
            image_path = st.session_state.captured_image['path']
            
            # Create placeholders for results
            progress_placeholder = st.empty()
            
            with progress_placeholder.container():
                st.info("🔄 Processing image with Gemini AI and generating segmentation masks...")
            
            # Run both processes in parallel
            def run_gemini_classification(img_path):
                """Wrapper function to get Gemini response and return it"""
                try:
                    from google import genai
                    from google.genai.errors import ServerError, APIError
                    import time
                    
                    client = genai.Client(api_key="YOUR_KEY")
                    
                    with open(img_path, "rb") as img_file:
                        image_bytes = img_file.read()
                    
                    max_retries = 5
                    for attempt in range(1, max_retries + 1):
                        try:
                            response = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=[
                                    {
                                        "role": "user",
                                        "parts": [
                                            {
                                                "text": (
                                                    "You are an expert waste management and recycling specialist. "
                                                    "Analyze this image carefully and provide a comprehensive waste classification analysis. "
                                                    "\n\nPlease provide your response in the following structured format:\n"
                                                    "STATUS: [Recyclable/Non-recyclable/Partially Recyclable]\n"
                                                    "OBJECT: [Name of the object]\n"
                                                    "MATERIAL: [Primary material(s) detected]\n"
                                                    "CONDITION: [Describe the object's condition - good/usable/damaged/contaminated]\n"
                                                    "RECYCLING_INSTRUCTIONS: [Step-by-step instructions on how to recycle this item properly]\n"
                                                    "ALTERNATIVES: [Alternative disposal methods if not recyclable, or reuse suggestions]\n"
                                                    "ENVIRONMENTAL_IMPACT: [Brief note on environmental impact if not recycled]\n"
                                                    "SPECIAL_NOTES: [Any special considerations, warnings, or tips for this item]\n"
                                                    "\n"
                                                    "Be specific and helpful. If multiple objects are visible, analyze the most prominent one. "
                                                    "If you cannot clearly identify the object, state that and provide best-guess analysis based on visible characteristics."
                                                )
                                            },
                                            {
                                                "inline_data": {
                                                    "mime_type": "image/jpeg",
                                                    "data": image_bytes
                                                }
                                            }
                                        ]
                                    }
                                ]
                            )
                            return response.text
                        except (ServerError, APIError, Exception) as e:
                            if attempt < max_retries:
                                wait = 2 ** attempt
                                time.sleep(wait)
                            else:
                                return f"❌ Error after {max_retries} attempts: {str(e)}"
                    return "❌ Failed to get response"
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            def run_segmentation(img_path, out_dir):
                """Run segmentation and return results"""
                try:
                    combined_overlay_path = extract_segmentation_masks(img_path, out_dir)
                    return {
                        'success': True,
                        'combined_overlay_path': combined_overlay_path,
                        'output_dir': out_dir,
                        'has_masks': combined_overlay_path is not None
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e),
                        'has_masks': False
                    }
            
            # Execute both processes in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                gemini_future = executor.submit(run_gemini_classification, image_path)
                segmentation_future = executor.submit(run_segmentation, image_path, output_dir)
                
                # Wait for both to complete
                gemini_result = gemini_future.result()
                segmentation_result = segmentation_future.result()
            
            # Update session state
            st.session_state.gemini_response = gemini_result
            st.session_state.segmentation_results = segmentation_result
            st.session_state.processing = False
            
            # Clear progress and show results
            progress_placeholder.empty()
            
            # Display results in right column will be updated automatically

with col2:
    st.header("📊 Analysis Results")
    
    # Display Gemini Response
    if st.session_state.gemini_response:
        with st.container():
            st.markdown("### 🔍 Comprehensive Waste Analysis")
            st.markdown("---")
            
            response_text = st.session_state.gemini_response
            
            # Parse structured response
            def parse_response(text):
                """Parse the structured Gemini response"""
                parsed = {}
                lines = text.split('\n')
                current_key = None
                current_value = []
                
                for line in lines:
                    line = line.strip()
                    if ':' in line and not line.startswith(' '):
                        # Save previous key-value
                        if current_key:
                            parsed[current_key] = '\n'.join(current_value).strip()
                        
                        # Extract new key-value
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            current_key = parts[0].strip()
                            current_value = [parts[1].strip()] if parts[1].strip() else []
                        else:
                            current_key = None
                            current_value = []
                    elif current_key and line:
                        current_value.append(line)
                    elif not current_key and line:
                        # Free-form text at the beginning
                        if 'FREE_FORM' not in parsed:
                            parsed['FREE_FORM'] = []
                        parsed['FREE_FORM'].append(line)
                
                # Save last key-value
                if current_key:
                    parsed[current_key] = '\n'.join(current_value).strip()
                
                return parsed
            
            parsed_response = parse_response(response_text)
            
            # Display Status
            status = parsed_response.get('STATUS', '')
            if status:
                status_lower = status.lower()
                if 'recyclable' in status_lower and 'non' not in status_lower and 'partially' not in status_lower:
                    st.success(f"✅ **Status: {status}**")
                elif 'partially' in status_lower:
                    st.warning(f"⚠️ **Status: {status}**")
                elif 'non-recyclable' in status_lower:
                    st.error(f"❌ **Status: {status}**")
                else:
                    st.info(f"📋 **Status: {status}**")
            
            # Display Object and Material in columns
            col_obj, col_mat = st.columns(2)
            with col_obj:
                if parsed_response.get('OBJECT'):
                    st.metric("Object", parsed_response['OBJECT'])
            with col_mat:
                if parsed_response.get('MATERIAL'):
                    st.metric("Material", parsed_response['MATERIAL'])
            
            # Display Condition
            if parsed_response.get('CONDITION'):
                condition = parsed_response['CONDITION']
                condition_lower = condition.lower()
                if 'good' in condition_lower or 'usable' in condition_lower:
                    st.info(f"🔵 **Condition:** {condition}")
                elif 'damaged' in condition_lower or 'contaminated' in condition_lower:
                    st.warning(f"🟡 **Condition:** {condition}")
                else:
                    st.caption(f"**Condition:** {condition}")
            
            # Display Recycling Instructions
            if parsed_response.get('RECYCLING_INSTRUCTIONS'):
                with st.expander("♻️ Recycling Instructions", expanded=True):
                    instructions = parsed_response['RECYCLING_INSTRUCTIONS']
                    # Format as bullet points if it's a list
                    if '\n' in instructions or '•' in instructions or '-' in instructions:
                        st.markdown(instructions)
                    else:
                        st.markdown(f"• {instructions}")
            
            # Display Alternatives
            if parsed_response.get('ALTERNATIVES'):
                with st.expander("💡 Alternatives & Reuse Options"):
                    st.markdown(parsed_response['ALTERNATIVES'])
            
            # Display Environmental Impact
            if parsed_response.get('ENVIRONMENTAL_IMPACT'):
                with st.expander("🌍 Environmental Impact"):
                    st.info(parsed_response['ENVIRONMENTAL_IMPACT'])
            
            # Display Special Notes
            if parsed_response.get('SPECIAL_NOTES'):
                with st.expander("📝 Special Notes"):
                    st.markdown(parsed_response['SPECIAL_NOTES'])
            
            # If response doesn't follow structure, show full response
            if not parsed_response or len(parsed_response) <= 1:
                with st.expander("📄 Full Analysis"):
                    st.text_area("", value=response_text, height=200, disabled=True, label_visibility="collapsed")
    
    # Display Segmentation Results
    if st.session_state.segmentation_results:
        st.markdown("### 🎨 Segmentation Results")
        st.markdown("---")
        
        if st.session_state.segmentation_results['success']:
            if st.session_state.segmentation_results.get('has_masks', False):
                combined_path = st.session_state.segmentation_results.get('combined_overlay_path')
                if combined_path and os.path.exists(combined_path):
                    st.success("✅ Masks Generated Successfully!")
                    
                    # Display combined overlay
                    overlay_img = PILImage.open(combined_path)
                    st.image(overlay_img, caption="Segmentation Overlay")
                    
                    # Count masks
                    output_dir = st.session_state.segmentation_results.get('output_dir', 'segmentation_outputs')
                    if os.path.exists(output_dir):
                        mask_files = [f for f in os.listdir(output_dir) if f.endswith('_mask.png')]
                        if mask_files:
                            st.info(f"📦 Detected **{len(mask_files)}** object(s)")
                else:
                    st.info("No overlay image available")
            else:
                st.warning("⚠️ No objects detected")
                st.caption("The image was processed but no objects were found for segmentation. Try capturing a clearer image with visible objects.")
        else:
            st.error(f"❌ Segmentation failed: {st.session_state.segmentation_results.get('error', 'Unknown error')}")
    else:
        st.info("👈 Capture an image and click 'Analyze Image' to see results here")

# Footer
st.markdown("---")
st.caption("💡 **Tip:** Start the live feed, position your object, then capture to analyze it.")
