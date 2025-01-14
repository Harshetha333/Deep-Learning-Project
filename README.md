# Real-Time Object Detection with YOLOv8

## Project Overview
This project focuses on developing a real-time object detection system using the YOLOv8 architecture. The goal is to accurately detect and localize objects in images using a trained model that can be deployed for real-time applications.

## Repository Structure
- `CODE/`: Contains all the source code.
  - `Main_code.ipynb`: Notebook with the model's training and validation process.
  - `deploy.py`: Deployment script for the model using Gradio.
  - `best.pt`: Trained model weights.
- `data/`: Folder for the dataset used in model training.
  - `images.zip`: Annotated images used for training the model.
- `docs/`: Documentation and additional resources.
  - `Final_report.pdf`: Detailed explanation of the methodology, architecture, and outcomes.
  - `deployment.mp4`: Demonstration of the model deployment.
- `README.md`: This file, containing project details and setup instructions.
- `requirements/`: Dependencies required for the project.

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/project-name.git
   ```
2. **Install Dependencies**
   Navigate to the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Explore the Notebook**
   Open `Main_code.ipynb` in a Jupyter environment to see the model's training and validation.
4. **Run the Deployment Script**
   Execute `deploy.py` to start the Gradio web interface and test the model in real-time.

## Usage
Upload an image through the Gradio interface, and the model will detect objects, highlighting them with bounding boxes and class labels.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your suggested changes.

## License
This project is open-sourced under the MIT license.

## Acknowledgements
Thanks to all contributors and researchers who have provided insights and tools for effective object detection using deep learning.

## Contact
- Sri Harshetha Amaravadi
- asriharshetha@gmail.com
- https://www.linkedin.com/in/sriharshethaamaravadi/
