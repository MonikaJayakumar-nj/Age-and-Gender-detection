# Age-and-Gender-detection

Achieved upto 95% accuracy using OpenCV's Deep Neural Network model to detect faces and predict age and gender from images, videos, or webcam feeds. It employs pre-trained models for face detection, gender classification, and age estimation. Detected faces are annotated with predicted gender (Male/Female) and age group (e.g., 25-32), with confidence scores displayed. The system supports real-time processing and saves the results for further use, making it suitable for applications in analytics, security, and personalization.

Usage
1. Detect Age and Gender from an Image
To detect age and gender from an image, execute the following command:

```bash
python gender_age.py --input path/to/image.jpg --output path/to/save/detected/images
