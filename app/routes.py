import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from .validation import BusinessValidationError, InternalServerError
from .mediapipe_handler import GestureRecognizer

api = Blueprint('api', __name__)
gesture_recognizer = GestureRecognizer()

# Define temp_dir here
temp_dir = os.path.join(os.path.dirname(__file__), '../temp')
os.makedirs(temp_dir, exist_ok=True)

@api.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image_file' not in request.files:
            raise BusinessValidationError(
                status_code=400,
                error_message="Image file not found. Ensure that multipart/form-data contains key - 'image_file'"
            )

        image_file = request.files['image_file']

        if image_file.filename == '':
            raise BusinessValidationError(
                status_code=400,
                error_message="No selected file"
            )

        # Validate file format (example: check for image file extensions)
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if not allowed_file(image_file.filename, allowed_extensions):
            raise BusinessValidationError(
                status_code=400,
                error_message="Unsupported file format. Supported formats: jpg, jpeg, png, gif"
            )

        filename = secure_filename(image_file.filename)
        temp_path = os.path.join(temp_dir, filename)
        image_file.save(temp_path)

        # Handle large file size
        # Example: check file size against a limit (adjust as per your requirement)
        max_file_size_mb = 10  # 10 MB limit
        if os.path.getsize(temp_path) > max_file_size_mb * 1024 * 1024:
            os.remove(temp_path)
            raise BusinessValidationError(
                status_code=413,
                error_message=f"File exceeds maximum size of {max_file_size_mb} MB"
            )

        result = gesture_recognizer.process_image(image_file=temp_path)
        os.remove(temp_path)

        # Handle recognition result
        if result is None:
            raise BusinessValidationError(
                status_code=404,
                error_message="No gesture recognized. Please try again with a different image."
            )

        # Example response format
        return jsonify({
            'handedness': result.handedness[0][0].category_name,
            'gesture': result.gestures[0][0].category_name
        })

    except BusinessValidationError as error:
        return error.response

    except IOError as error:
        print(f"IOError: {error}")
        raise InternalServerError(error_message='Internal Server Error')

    except Exception as error:
        print(f"Unhandled Exception: {error}")
        raise InternalServerError(error_message='Internal Server Error')

# Helper function to check allowed file extensions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
