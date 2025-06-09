# MLX8 Week 1

## Feature Fusion

The first project will focus on **predicting upvotes for Hacker News posts**. The primary goal is to build predictive models by leveraging domain-specific word embeddings trained on English Wikipedia and fine-tuned from Hacker News titles. This project combines the practical aspects of machine learning with exploratory data analysis (EDA). The dataset contains over 12 million Hacker News posts. Through this approach, you will gain hands-on experience with data preparation, feature extraction, and the iterative process of model tuning.

To support the technical objectives, we will explore two fundamental architectures for word embedding:

- **Continuous Bag of Words (CBOW)**
- **Skip-gram**

Both are foundational techniques in the Word2Vec framework. To effectively incorporate both author and website features into the predictive models, fusion architectures such as **Early Fusion** and **Late Fusion** will be explored.

For deployment and scaling, we will introduce simple DevOps practices such as containerization with Docker, enabling reproducible environments and easy integration into larger workflows. Additionally, tools like PyTorch and GPUs will be employed for model training and experimentation.

---

### What you will build in practice

- Text Preprocessing Pipeline
- Custom Word Embeddings for Hacker News
- Exploratory Data Analysis using SQL, Pandas, and Python
- Predictive Model for Upvotes using Fusion Architectures
- Model Deployment with Docker and FastAPI

---

### Tools and Libraries You Will Use

- **PyTorch**
- **Docker**
- **Python**
- **systemd**
- **FastAPI**
- **Jupyter**