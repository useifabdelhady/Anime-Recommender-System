# Anime Recommendation System

This is a hybrid anime recommendation system that combines user-based and content-based filtering to provide personalized anime recommendations.

## Project Structure

- `application.py`: Flask web application that serves the recommendation system
- `pipeline/`: Contains the prediction pipeline for generating recommendations
- `src/`: Source code for data processing and model training
- `utils/`: Helper functions for the recommendation system
- `config/`: Configuration files including paths
- `templates/`: HTML templates for the web interface
- `static/`: CSS and other static files
- `artifacts/`: Model weights and processed data (not included in repository)

## Deployment to Hugging Face Spaces

### Prerequisites

1. A Hugging Face account
2. Git installed on your local machine
3. Docker installed on your local machine (for local testing)

### Steps to Deploy

1. **Create a new Space on Hugging Face**

   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click on "Create a new Space"
   - Choose a name for your Space
   - Select "Docker" as the SDK
   - Choose visibility (Public or Private)
   - Click "Create Space"

2. **Clone the Space repository**

   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```

3. **Copy your project files to the cloned repository**

   Copy all files from this project to the cloned repository, including the Dockerfile.

4. **Add model artifacts**

   Make sure to include all necessary model artifacts in the `artifacts/` directory:
   - Processed data files in `artifacts/processed/`
   - Model weights in `artifacts/weights/`
   - Trained model in `artifacts/model/`

5. **Commit and push to Hugging Face**

   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```

6. **Monitor the build process**

   - Go to your Space on Hugging Face
   - Click on "Settings" and then "Repository"
   - Check the build logs to ensure everything is working correctly

7. **Access your deployed application**

   Once the build is complete, you can access your application at:
   `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

## Local Testing with Docker

Before deploying to Hugging Face, you can test your Docker setup locally:

```bash
# Build the Docker image
docker build -t anime-recommender .

# Run the container
docker run -p 5000:5000 anime-recommender
```

Then access the application at http://localhost:5000

## Important Notes

- Make sure all required model artifacts are included in your repository or are generated during the build process
- The application expects certain files to exist in the artifacts directory as specified in `config/paths_config.py`
- The Dockerfile is configured to run the application on port 5000