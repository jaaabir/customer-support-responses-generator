# T5 Customer Support Response Generation

This project involves fine-tuning a T5 model to generate automated responses for customer support queries using the dataset from [Kaludi/Customer-Support-Responses](https://huggingface.co/datasets/Kaludi/Customer-Support-Responses).

## Overview

The objective is to build a model capable of generating meaningful and accurate responses to customer queries. The T5 model, pre-trained by Google, has been fine-tuned on a dataset specific to customer support to achieve this task. This project encompasses data preprocessing, model training, evaluation using the BLEU metric, and response generation. 

## How the Model Works

The model is trained using a sequence-to-sequence approach where the input is a customer query and the output is the generated response. A specific prefix "Assure the customer and provide specific help" is added to each query to help the model understand the context and the task it needs to perform.

## Demo

Experience the model in action through [Streamlit app](https://streamlit.com).

## Setup Instructions

### Prerequisites

Ensure you have the following libraries installed after cloning the repository:

```bash
# Create and activate a virtual environment using venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt 

# Alternatively, using conda
conda create --name <env-name> --file requirements.txt
conda activate <env-name>
```

### Running the Notebook

Execute the model_training.ipynb notebook cell by cell to ensure proper functionality.

### Modifying Hyperparameters

To adjust the model parameters, edit the hyperparameters.py file and rerun the notebook for training.


## Conclusion

Most models trained with 5-10 epochs performed well. However, the model using the latest pretext specified in hyperparameters.py yielded the best results so far.

<img src="assets\eval_bleu.svg" alt="Evaluation BLEU score" style="width:600px;height:400px;">
<img src="assets\eval_loss.svg" alt="Evaluation Loss" style="width:600px;height:400px;">
