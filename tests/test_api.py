import torch
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.dog_breed_classifier.api import app

client = TestClient(app)


@patch("src.dog_breed_classifier.api.Image.open")
def test_predict_valid_image(mock_open):
    """
    Test the /predict/ endpoint with a simulated valid image.
    """
    # Mock the image loading
    mock_image = MagicMock()
    mock_open.return_value = mock_image

    # Mock the transformation pipeline
    with patch("src.dog_breed_classifier.api.transform") as mock_transform:
        # Simulate the transformation output as a valid torch tensor
        mock_transform.return_value = torch.zeros(1, 3, 224, 224)

        # Mock the model's output
        with patch("src.dog_breed_classifier.api.model") as mock_model:
            # Simulate the model's output as a tensor with logits
            mock_model.return_value = torch.tensor([[0.1, 0.7, 0.2]])
            
            # Send a POST request with a fake file
            response = client.post(
                "/predict/",
                files={"file": ("fake.jpg", b"fake image content", "image/jpeg")}
            )

            # Assertions
            assert response.status_code == 200, response.text  # Include response text for debugging
            response_json = response.json()
            assert "predicted_breed" in response_json
            assert isinstance(response_json["predicted_breed"], str)  # Ensure the result is a string


def test_predict_invalid_file_type():
    """
    Test the /predict/ endpoint with a non-image file.
    """
    response = client.post(
        "/predict/",
        files={"file": ("invalid.txt", b"this is not an image", "text/plain")}
    )

    # Assertions
    assert response.status_code == 400
    response_json = response.json()
    assert response_json["detail"] == "File must be an image (png, jpg, jpeg)."


def test_predict_missing_file():
    """
    Test the /predict/ endpoint without uploading any file.
    """
    response = client.post("/predict/")

    # Assertions
    assert response.status_code == 422, response.text
    response_json = response.json()
    assert "detail" in response_json
    assert response_json["detail"][0]["msg"] == "Field required"  # Capitalized
