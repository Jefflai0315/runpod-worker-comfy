import runpod
from runpod.serverless.utils import rp_upload
import json
import base64
import numpy as np
import cv2
import os
import torch
import urllib.request
import logging
from io import BytesIO
from typing import Dict, Optional, Union, Tuple, Any

from PIL import Image
from ultralytics import YOLO, FastSAM
from rembg import remove
import requests
import time

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model paths
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolo11m.pt")
SAM_MODEL_PATH = os.environ.get("SAM_MODEL_PATH", "FastSAM-x.pt")

# Load models
yolo_model = YOLO(YOLO_MODEL_PATH)
sam_model = FastSAM(SAM_MODEL_PATH)

# Constants for ComfyUI
COMFY_API_AVAILABLE_INTERVAL_MS = 50
COMFY_API_AVAILABLE_MAX_RETRIES = 500
COMFY_POLLING_INTERVAL_MS = os.environ.get("COMFY_POLLING_INTERVAL_MS", 250)
COMFY_POLLING_MAX_RETRIES = os.environ.get("COMFY_POLLING_MAX_RETRIES", 500)
COMFY_HOST = "127.0.0.1:8188"

# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"


# Utility functions
def base64_to_image(base64_string: str) -> Image.Image:
    """Convert a Base64-encoded string to a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))


def decode_image(image_base64: str) -> np.ndarray:
    """Decode a Base64-encoded string to an RGB NumPy array."""
    image_data = base64.b64decode(image_base64)
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_image(image: Union[np.ndarray, Image.Image]) -> str:
    """Encode a NumPy array or PIL Image to a Base64-encoded PNG string."""
    if isinstance(image, Image.Image):
        with BytesIO() as output:
            image.save(output, format="PNG")
            return base64.b64encode(output.getvalue()).decode("utf-8")
    else:
        _, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")


def get_largest_bounding_box(detections: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest bounding box from a list of bounding boxes."""
    if not len(detections):
        return None
    boxes_areas = [(box, (box[2] - box[0]) * (box[3] - box[1])) for box in detections]
    largest_box, _ = max(boxes_areas, key=lambda x: x[1])
    return largest_box


def detect_humans(image: np.ndarray) -> np.ndarray:
    """Detect humans in an image using YOLO and return their bounding boxes."""
    yolo_results = yolo_model.predict(image)
    human_detections = []

    for result in yolo_results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
        logger.info(f"Detected classes: {class_ids}")

        # Filter for "person" class (class ID 0 for COCO dataset)
        for box, class_id in zip(boxes, class_ids):
            if int(class_id) == 0:
                human_detections.append(box)
    return np.array(human_detections)


def apply_mask_to_image(
    image: np.ndarray, mask_data: Any, background_color: Optional[str] = None
) -> Image.Image:
    """Apply a mask to an image and optionally change the background color."""
    original_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2RGBA))
    original_array = np.array(original_image)

    if hasattr(mask_data, "masks"):
        mask_array = mask_data.masks.data.numpy().transpose(1, 2, 0)
    else:
        mask_array = np.array(mask_data)

    if mask_array.ndim == 3 and mask_array.shape[-1] > 1:
        # Combine multiple channels by taking the maximum
        mask_array = mask_array.max(axis=-1)

    if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
        mask_array = mask_array.squeeze(-1)

    # Ensure the mask has the correct dimensions (resize if necessary)
    if mask_array.shape[:2] != original_array.shape[:2]:
        mask_array = cv2.resize(
            mask_array, (original_array.shape[1], original_array.shape[0])
        )
        logger.debug("Resized mask array shape: %s", mask_array.shape)

    # Create a binary alpha channel based on mask
    alpha_channel = (mask_array > 0.5).astype(np.uint8) * 255
    original_array[:, :, 3] = alpha_channel

    final_image = Image.fromarray(original_array, mode="RGBA")
    if background_color:
        background = Image.new("RGBA", original_image.size, background_color)
        final_image = Image.alpha_composite(background, final_image)
    return final_image


def remove_background(image_base64: str, background_color: Optional[str]) -> str:
    """Remove background from an image using YOLO and FastSAM or Rembg."""
    try:
        start_time = time.time()
        image = decode_image(image_base64)
        logger.info("Applying background removal process...")
        human_detections = detect_humans(image)

        if not human_detections.any():
            logger.warning("No humans detected. Using the full image as a fallback.")
            height, width, _ = image.shape
            box = [0, 0, width, height]  # Use the full image as a fallback box
        else:
            box = get_largest_bounding_box(human_detections)
            logger.info(f"Largest bounding box: {box}")
        # Process with FastSAM
        image_pil = Image.fromarray(image)
        image_pil.save("temp_input.jpg")
        masks = sam_model(
            "temp_input.jpg", bboxes=box, labels=[1], texts="a portrait of a person"
        )
        result_image = apply_mask_to_image(image, masks[0], background_color)
        logger.debug(
            f"Segmented image process took {time.time() - start_time:.2f} seconds."
        )
        return encode_image(result_image)
    except Exception as e:
        logger.error(f"Error removing background: {e}")
        try:
            logger.warning("Using fallback method - rembg")
            image_pil = Image.fromarray(image)
            image_removed = remove(image_pil)
            logger.debug(
                f"Segmented image process took {time.time() - start_time:.2f} seconds."
            )
            return encode_image(image_removed)
        except Exception as err:
            logger.error(f"Error removing background: {err}")
            return None


def validate_input(job_input: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Validates the input for the handler function.
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided
    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys.",
            )
    upload_to_s3 = job_input.get("upload_to_s3", False)
    background_color = job_input.get("background", "#FFFFFF")
    steps = job_input.get(
        "steps", ["remove_background", "generate_portrait", "remove_add_background"]
    )

    return {
        "workflow": workflow,
        "images": images,
        "s3": upload_to_s3,
        "background": background_color,
        "steps": steps,
    }, None


def check_server(url: str, retries: int = 500, delay: int = 50) -> bool:
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    for i in range(retries):
        try:
            response = requests.get(url)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"runpod-worker-comfy - API is reachable")
                return True
        except requests.RequestException as e:
            # If an exception occurs, the server may not be ready
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"runpod-worker-comfy - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


def upload_images(images: list) -> Dict[str, Any]:
    """
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.
        server_address (str): The address of the ComfyUI server.

    Returns:
        list: A list of responses from the server for each image upload.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    logger.info(f"runpod-worker-comfy - image(s) upload")

    for image_data in images:
        name = image_data["name"]
        image_base64 = image_data["image"]
        blob = base64.b64decode(image_base64)
        files = {
            "image": (name, BytesIO(blob), "image/png"),
            "overwrite": (None, "true"),
        }
        try:
            response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)
            if response.status_code == 200:
                responses.append(f"Successfully uploaded {name}")
            else:
                upload_errors.append(f"Error uploading {name}: {response.text}")
        except requests.RequestException as e:
            upload_errors.append(f"Error uploading {name}: {e}")

    if upload_errors:
        logger.error("Some images failed to upload")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    logger.info("Image(s) uploaded successfully.")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }


def queue_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed

    Returns:
        dict: The JSON response from ComfyUI after processing the workflow
    """

    # The top level element "prompt" is required by ComfyUI
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        error_details = e.read().decode()
        logger.error(f"Error in queue_workflow: {e}, Response: {error_details}")
        raise


def get_history(prompt_id: str) -> Dict[str, Any]:
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as response:
        return json.loads(response.read())


def base64_encode(img_path: str) -> str:
    """
    Returns base64 encoded image.

    Args:
        img_path (str): The path to the image

    Returns:
        str: The base64 encoded image
    """
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string


def list_directory_contents(directory: str) -> Union[list, str]:
    try:
        return os.listdir(directory)
    except Exception as e:
        return str(e)


def process_output_images(
    outputs: Dict[str, Any], job: Dict[str, Any], upload_to_s3: bool
) -> Dict[str, Any]:
    """
    This function takes the "outputs" from image generation and the job ID,
    then determines the correct way to return the image, either as a direct URL
    to an AWS S3 bucket or as a base64 encoded string, depending on the
    environment configuration.

    Args:
        outputs (dict): A dictionary containing the outputs from image generation,
                        typically includes node IDs and their respective output data.
        job_id (str): The unique identifier for the job.

    Returns:
        dict: A dictionary with the status ('success' or 'error') and the message,
              which is either the URL to the image in the AWS S3 bucket or a base64
              encoded string of the image. In case of error, the message details the issue.

    The function works as follows:
    - It first determines the output path for the images from an environment variable,
    #   defaulting to "/comfyui/output" if not set.
    - It then iterates through the outputs to find the filenames of the generated images.
    - After confirming the existence of the image in the output folder, it checks if the
      AWS S3 bucket is configured via the BUCKET_ENDPOINT_URL environment variable.
    - If AWS S3 is configured, it uploads the image to the bucket and returns the URL.
    - If AWS S3 is not configured, it encodes the image in base64 and returns the string.
    - If the image file does not exist in the output folder, it returns an error status
      with a message indicating the missing image file.
    """

    # The path where ComfyUI stores the generated images
    COMFY_OUTPUT_PATH = os.environ.get("COMFY_OUTPUT_PATH", "/comfyui/output")
    output_file_path = None

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for image_data in node_output["images"]:
                if image_data["type"] == "output":
                    output_file_path = os.path.join(
                        image_data["subfolder"], image_data["filename"]
                    )
                    break
            if output_file_path:
                break

    if not output_file_path:
        logger.error("No output images found in workflow results.")
        return {"status": "error", "message": "No output images found"}

    full_image_path = os.path.join(COMFY_OUTPUT_PATH, output_file_path)
    if os.path.exists(full_image_path):
        if upload_to_s3:
            bucket_url = os.environ.get("BUCKET_ENDPOINT_URL")
            if bucket_url:
                # Assume `rp_upload` handles S3 uploading and returns a URL
                infographic_id = (
                    job["input"]["images"][0]["filename"].split("/", 1)[0]
                    if job["input"]["images"]
                    else "output"
                )
                s3_url = rp_upload.upload_image(
                    job["id"], full_image_path, 0, None, infographic_id
                )
                logger.info(f"Image uploaded to AWS S3: {s3_url}")
                return {"status": "success", "message": s3_url}
            else:
                logger.warning(
                    "S3 bucket endpoint not configured. Returning base64 data."
                )
                encoded_image = base64_encode(full_image_path)
                return {"status": "success", "message": encoded_image}
        else:
            # Return Base64-encoded image
            encoded_image = base64_encode(full_image_path)
            logger.info("Returning base64-encoded output image.")
            return {"status": "success", "message": encoded_image}
    else:
        current_dir = os.getcwd()
        error_message = (
            f"The image does not exist at the expected path: {full_image_path}\n"
            f"Current working directory: {current_dir}\n"
            f"COMFY_OUTPUT_PATH: {COMFY_OUTPUT_PATH}\n"
            f"Directory listing: {list_directory_contents(current_dir)}"
        )
        logger.error(error_message)
        return {"status": "error", "message": error_message}


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    The main function that handles a job of generating an image.

    This function validates the input,
    processes the steps,
    removes background,
    sends a prompt to ComfyUI for processing,
    polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]
    start_time = time.time()
    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    # Extract validated data
    workflow = validated_data["workflow"]
    images = validated_data.get("images")
    upload_to_s3 = validated_data.get("s3")
    background_color = validated_data.get("background")
    steps = validated_data.get("steps")
    max_retries = 3
    image_base64 = images[0]["image"]

    if not image_base64:
        logger.error("No image data provided.")
        return {"error": "No image data provided."}

    # Results dictionary to store statuses of each step
    step_results = {}

    for step in steps:
        for attempt in range(max_retries):
            try:
                if step == "remove_add_background":
                    logger.info("Performing 'remove_add_background' step.")
                    image_base64 = remove_background(image_base64, background_color)
                    step_results["remove_add_background"] = {
                        "status": "success",
                        "image": image_base64,
                    }
                    break  # Exit the retry loop on success

                elif step == "remove_background":
                    logger.info("Performing 'remove_background' step.")
                    image_base64 = remove_background(image_base64, None)
                    step_results["remove_background"] = {
                        "status": "success",
                        "image": image_base64,
                    }
                    break  # Exit the retry loop on success

                elif step == "generate_portrait":
                    logger.info("Performing 'generate_portrait' step.")
                    # Ensure that the ComfyUI API is available
                    if not check_server(
                        f"http://{COMFY_HOST}",
                        COMFY_API_AVAILABLE_MAX_RETRIES,
                        COMFY_API_AVAILABLE_INTERVAL_MS,
                    ):
                        return {"error": "ComfyUI API is not reachable."}

                    # Upload images if provided
                    upload_result = upload_images(images)
                    if upload_result["status"] == "error":
                        step_results["generate_portrait"] = upload_result
                        break

                    # Queue the workflow in ComfyUI
                    try:
                        queued_workflow = queue_workflow(workflow)
                        prompt_id = queued_workflow["prompt_id"]
                        logger.info(f"Workflow queued with ID {prompt_id}")
                    except Exception as e:
                        step_results["generate_portrait"] = {
                            "error": f"Error queuing workflow: {str(e)}"
                        }
                        break

                    # Poll until image generation is complete
                    for _ in range(COMFY_POLLING_MAX_RETRIES):
                        history = get_history(prompt_id)
                        if prompt_id in history and "outputs" in history[prompt_id]:
                            images_result = process_output_images(
                                history[prompt_id]["outputs"], job, upload_to_s3
                            )
                            if images_result["status"] == "success":
                                image_base64 = images_result["message"]
                                step_results["generate_portrait"] = {
                                    "status": "success",
                                    "image": image_base64,
                                }
                            else:
                                step_results["generate_portrait"] = images_result
                            break
                        time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
                    else:
                        step_results["generate_portrait"] = {
                            "error": "Max retries reached while waiting for image generation"
                        }
                    break  # Exit the retry loop

                else:
                    # Unknown step
                    logger.warning(f"Unknown step: {step}")
                    step_results[step] = {
                        "status": "skipped",
                        "message": "Unknown step",
                    }
                    break

            except Exception as e:
                logger.error(f"Error in step '{step}' on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    step_results[step] = {
                        "error": f"Failed at step '{step}' after {max_retries} attempts"
                    }
                    return (
                        step_results  # Return early if a step fails after all retries
                    )

    # Compile final result
    final_result = {
        **step_results,
        "refresh_worker": REFRESH_WORKER,
    }

    logger.info("Workflow completed in: ", time.time() - start_time, " seconds")
    if os.path.exists("temp_input.jpg"):
        os.remove("temp_input.jpg")
    return final_result


# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
