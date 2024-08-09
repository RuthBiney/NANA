# Wine Quality Prediction API

## Project Overview

This project is a Wine Quality Prediction API built using FastAPI. The API allows users to predict the quality of wine based on several chemical properties. The machine learning model used for prediction is a Multi-Layer Perceptron (MLP) model trained on the Wine Quality dataset. The project is containerized using Docker, making it easy to deploy and run in various environments.

## Features

- **Wine Quality Prediction:** Provides an API endpoint to predict the quality of wine based on input features.
- **Model Retraining:** Offers an endpoint to retrain the model with new data.
- **Dockerized:** The application is containerized for easy deployment and scalability.

## Technologies Used

- **FastAPI:** A modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.
- **scikit-learn:** A machine learning library used to create and train the model.
- **Docker:** Containerization platform used to package the application and its dependencies.

## Project Structure

Project_name/
├── app/
│ ├── main.py # Main FastAPI application
│ ├── requirements.txt # Python dependencies
│ ├── mlp_model.pkl # Pre-trained machine learning model
│ ├── winequality-red.csv # Red wine dataset for retraining
│ ├── winequality-white.csv # White wine dataset for retraining
├── Dockerfile # Docker configuration file
├── README.md # Project documentation

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Docker (for running the containerized application)

### Running Locally

1. **Clone the Repository:**

   ```sh
   git clone https://github.com/RuthBiney/NANA.git
   cd "Current working directory from your end"
   ```

### Create a Virtual Environment:

python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

### Install Dependencies:

pip install --upgrade pip
pip install -r app/requirements.txt

### Run the Application:

uvicorn app.main:app --reload

### Access the API Documentation:

Open your browser and navigate to http://127.0.0.1:8000/docs to view the interactive API documentation.

### Running with Docker

### Build the Docker Image:

docker build -t mlapp:latest .

### Run the Docker Container:

docker run -d --name mlapp-instance -p 8000:8000 mlapp:latest

### Access the API:

Open your browser and navigate to http://localhost:8000/docs to access the interactive API documentation.

### API Endpoints

Description: Predicts the quality of wine based on input features.
Request Body:
{
"fixed_acidity": 7.4,
"volatile_acidity": 0.7,
"citric_acid": 0.0,
"residual_sugar": 1.9,
"chlorides": 0.076,
"free_sulfur_dioxide": 11.0,
"total_sulfur_dioxide": 34.0,
"density": 0.9978,
"pH": 3.51,
"sulphates": 0.56,
"alcohol": 9.4
}

### /retrain/ [POST]

Description: Retrains the model with new wine data (red and white wine datasets).
{
"message": "Model retrained successfully"
}
