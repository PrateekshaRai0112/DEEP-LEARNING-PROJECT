# DEEP-LEARNING-PROJECT



## Sentiment Classification Using PyTorch

This project demonstrates a simple yet powerful Natural Language Processing (NLP) model built using PyTorch. The main goal is to classify text data into positive or negative sentiment, which is a common task in opinion mining, customer feedback analysis, and chatbot applications.

The project walks through the complete workflow of building a deep learning model: loading the dataset, preprocessing the text, building the neural network model, training it using a loss function and optimizer, evaluating performance, and finally visualizing results like training loss and predictions.

## About the Dataset

The dataset used in this project is a small manually curated CSV file named sample_data.csv, which contains 10 sentences labeled as either positive (1) or negative (0). 

Here’s an example of the data:
Text	Label

"I love this product!"	1

"This is the worst experience."	0

"Absolutely fantastic service!"	1

"Terrible. Wouldn't recommend."	0

Despite its small size, this dataset allows for testing the full pipeline of an NLP classification model in a manageable and easily understandable way.

## Technologies Used
The project is implemented using the following Python libraries:
`PyTorch` - Deep learning framework used to build and train the model.
`pandas` - For data loading and basic manipulation.
`scikit-learn` - For train-test split and feature extraction using CountVectorizer.
`matplotlib` - For visualizing the training loss over epochs.

## How the Project Works

1. Text Preprocessing
The sentences in the dataset are converted into numerical format using Bag-of-Words (CountVectorizer). This transformation allows the model to interpret textual data as numerical features.
3. Model Architecture
The model is a simple 2-layer fully connected feedforward neural network:
- Input Layer: Takes in the BoW features.
-	Hidden Layer: 64 neurons with ReLU activation.
-	Output Layer: 2 neurons for binary classification (positive or negative).
This is not an LSTM model because the dataset is very small. However, this structure is sufficient to demonstrate the fundamentals of text classification using PyTorch.
3. Training Process
•	Loss Function: CrossEntropyLoss (appropriate for multi-class classification tasks).
•	Optimizer: Adam optimizer, with a learning rate of 0.01.
•	Epochs: 20 iterations over the training set.
•	During each epoch, the model calculates the loss and updates weights accordingly.
4. Evaluation
After training, the model is tested on a small test set (20% split from the dataset). It outputs predictions and shows whether the model predicted "Positive" or "Negative" for each sample.
5. Visualization
•	The training loss across 20 epochs is plotted and saved as training_loss.png.
•	This helps visualize the convergence of the model and indicates if the model is learning.

 Project Structure
nlp_sentiment_analysis
sample_data.csv             
sentiment_lstm.py           
training_loss.png           
README.md                   

 How to Run the Project
1.	Install Required Packages
pip install torch matplotlib pandas scikit-learn
2.	Run the Training Script
      python sentiment_lstm.py
3.	Check the Output
o	Console will print loss per epoch and predictions.
o	A plot of training loss will be saved as training_loss.png.

Why This Project Matters
Even though the dataset is small and the model is simple, this project captures the core steps of building an NLP classification pipeline using PyTorch. These steps include:
•	Handling text data
•	Vectorizing features
•	Designing and training a model
•	Visualizing performance
•	Making predictions on unseen data
This structure is scalable. In the future, you can replace the dataset with real-world data and swap the model with an LSTM or Transformer-based architecture to solve more complex problems.
 Conclusion
This project serves as a beginner-friendly and complete walkthrough for building a text classification model using PyTorch. From raw sentences to model predictions and visual insights, it highlights the end-to-end workflow of NLP projects. It’s a great starting point to dive deeper into deep learning and natural language processing.

