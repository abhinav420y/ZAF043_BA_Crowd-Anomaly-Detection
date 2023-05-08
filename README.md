# ZAF043_BA_Crowd-Anomaly-Detection

## Overview
Crowd anomaly detection is a type of machine learning model that is designed to detect unusual behavior in crowds of people. This can be useful for a variety of applications, including security, public safety, and crowd management.

The basic idea behind a crowd anomaly detection model is to use machine learning algorithms to analyze video footage or other types of sensor data in order to identify patterns of behavior that are outside the norm. These patterns could include things like sudden movements, large crowds gathering in unusual places, or groups of people moving in ways that are inconsistent with normal crowd behavior.

## Methodology
- Collect Data: The first step in building a crowd anomaly detection model is to collect a large dataset of crowd behavior. This data can be collected using sensors such as cameras, microphones, or other IoT devices.

- Preprocess Data: Once the data is collected, it needs to be preprocessed to remove any noise or outliers that may interfere with the analysis. This can involve tasks such as filtering, smoothing, and normalization.

- Data Annotation: In order to train a deep learning model, the dataset needs to be labeled with appropriate annotations that indicate normal and abnormal behavior. This can be done manually by human annotators or using automated tools.

- Split Data: The labeled dataset is then split into training, validation, and testing sets. The training set is used to train the model, the validation set is used to optimize the hyperparameters of the model, and the testing set is used to evaluate the performance of the model.

- Build Model: Once the data is preprocessed and split, the next step is to build a deep learning model. Convolutional Neural Networks (CNN) are commonly used for crowd anomaly detection tasks due to their ability to learn spatial features from images or videos.

- Train Model: The model is then trained using the labeled training dataset. This involves repeatedly presenting the model with input data and adjusting the model's parameters to minimize the error between the predicted outputs and the ground truth.

- Validate Model: After the model is trained, it is validated on the validation dataset to ensure that it is not overfitting to the training data. This involves monitoring the model's performance on the validation set and adjusting the model's hyperparameters as needed.

- Test Model: Finally, the model is tested on the testing dataset to evaluate its performance on new, unseen data. The performance metrics of the model, such as accuracy, precision, recall, F1-score, etc., are computed and analyzed.

- Deploy Model: If the model performs well on the testing dataset, it can be deployed in the real world to detect crowd anomalies in real-time. This involves integrating the model with sensors and other IoT devices and developing a user interface for the end-users to visualize and analyze the model's outputs.

## Dataset
- [Avenue Dataset](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)

## Papers
- [Abnormal Event Detection in Videos using Spatiotemporal Autoencoder](https://arxiv.org/abs/1701.01546)

## Instructions
To use the given model follow the instructions below:
1. Clone the repository using the following command
```
git clone https://github.com/abhinav420y/ZAF043_BA_Crowd-Anomaly-Detection
```
2. Install the requirements using the below command
```
pip install -r requirements.txt
```
3. Run 'load_predict.py' to run the model on given test.avi.