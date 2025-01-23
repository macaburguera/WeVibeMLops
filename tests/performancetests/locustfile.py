import os
from locust import HttpUser, between, task

class DogBreedClassifierUser(HttpUser):
    """
    A Locust user class that simulates user interaction with the dog breed classification API.
    """

    wait_time = between(1, 2)  # Simulates a wait time between tasks

    @task
    def predict_breed(self):
        """
        A task that sends a POST request to the /predict/ endpoint with a mock image.
        """
        # Create a dummy image file in memory
        mock_image = ("mock_image.jpg", b"fake image content", "image/jpeg")

        # Send the POST request
        response = self.client.post("/predict/", files={"file": mock_image})

        # Optionally log the response or check for success
        if response.status_code == 200:
            print("Prediction succeeded:", response.json())
        else:
            print("Prediction failed:", response.text)
