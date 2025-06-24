"""
Multi-Camera Face Detection and Emotion Recognition System

This program processes multiple RTSP camera streams simultaneously,
performing face detection, recognition, and emotion analysis.

Library Functions Usage Guide:

OpenCV (cv2) Functions:
- cv2.VideoCapture(url, cv2.CAP_FFMPEG): Creates a video capture object for streaming
- cv2.CAP_FFMPEG: Backend flag for using FFMPEG for video capture
- cap.isOpened(): Checks if video capture is successfully initialized
- cap.set(cv2.CAP_PROP_BUFFERSIZE, size): Sets the internal buffer size
- cap.read(): Reads a frame from video stream, returns (success_flag, frame)
- cv2.resize(frame, size): Resizes an image to specified dimensions
- cv2.cvtColor(frame, cv2.COLOR_BGR2RGB): Converts color space from BGR to RGB
- cv2.rectangle(frame, pt1, pt2, color, thickness): Draws a rectangle on the frame
- cv2.putText(frame, text, org, font, scale, color, thickness): Puts text on the frame
- cv2.imshow(window_name, frame): Displays a frame in a window
- cv2.waitKey(delay): Waits for a keyboard event for specified milliseconds
- cv2.destroyWindow(window_name): Closes a specific window
- cv2.destroyAllWindows(): Closes all OpenCV windows
- cv2.namedWindow(name, cv2.WINDOW_NORMAL): Creates a window with resizable property
- cv2.resizeWindow(name, width, height): Sets window size
- cv2.dnn.readNetFromCaffe(): Loads a pre-trained Caffe model
- cv2.dnn.blobFromImage(): Prepares image for neural network input

face_recognition Functions:
- face_recognition.load_image_file(path): Loads an image file for face processing
- face_recognition.face_encodings(image): Generates face encodings from image
- face_recognition.face_distance(known_encodings, face_encoding): Calculates face similarity distances

DeepFace Functions:
- DeepFace.analyze(image, actions=['emotion']): Analyzes face for emotions
  Returns dictionary with:
  - dominant_emotion: Most probable emotion
  - emotion: Dictionary of emotion probabilities

numpy (np) Functions:
- np.argmin(array): Returns index of minimum value in array
- np.any(condition): Tests if any array element is True
- np.isnan(array): Checks for "Not a Number" values
- np.isinf(array): Checks for infinite values

os Functions:
- os.path.exists(path): Checks if a file/directory exists
- os.path.join(path1, path2): Joins path components
- os.listdir(path): Lists contents of a directory
- os.path.isdir(path): Checks if path is a directory

logging Functions:
- logging.basicConfig(): Configures logging system
- logging.error(msg): Logs error message
- logging.warning(msg): Logs warning message
- logging.info(msg): Logs informational message
- logging.debug(msg): Logs debug message

asyncio Functions:
- asyncio.create_task(coro): Creates a task for async execution
- asyncio.gather(*tasks): Waits for multiple async tasks to complete
- asyncio.sleep(seconds): Asynchronous sleep
- asyncio.run(main()): Runs the main async function

csv Functions:
- csv.writer(file): Creates a CSV writer object
- writer.writerow(list): Writes a row to CSV file

datetime Functions:
- datetime.now(): Gets current date and time
- datetime.isoformat(): Converts datetime to ISO format string
"""