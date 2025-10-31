import cv2
import time
from google import genai
from google.genai.errors import ServerError, APIError

# Initialize Gemini client
client = genai.Client(api_key="AIzaSyAgvzUpebcc-UFxcKjwWcdc7sjGNdVMvP8")

def capture_frame(filename="frame.jpg"):
    """Capture one frame from the webcam and save it."""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return None

    print("Capturing frame...")
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if not ret:
        print("Error: Failed to capture image.")
        return None

    cv2.imwrite(filename, frame)
    print(f"Frame saved as {filename}")
    return filename


def classify_recyclability(image_path, max_retries=5):
    """Send image to Gemini to classify recyclability, with retry logic."""
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

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
                                    "Don't be partial, be objective and give the best answer."
                                    "You are an image classification expert. " 
                                    "Remember that plastics, metals, and paper are recyclable and not a waste." 
                                    "Look carefully at this image and determine if the visible object is recyclable or not. "
                                    "Answer with only one word: 'Recyclable' or 'Non-recyclable'. "
                                    "Then state the object's name and material. "
                                    "If the object is a pen, headphones, or watch, classify it as 'Recyclable'."
                                    "Remember that plastics, metals, and paper are recyclable." 
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

            print("\nGemini response:")
            print(response.text)
            return  # success — exit the function

        except (ServerError, APIError, Exception) as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print("❌ All retry attempts failed. Gemini service may be overloaded. Try again later.")


if __name__ == "__main__":
    frame_file = capture_frame()
    if frame_file:
        classify_recyclability(frame_file)
