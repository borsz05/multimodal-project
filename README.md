# BERT–CLIP Dual Encoder for Image–Text Retrieval

## Authors
- **Tóth Bertalan György**
- **Vincze Balázs Iván**

---

## Project Overview
This project implements a **CLIP-like multimodal image–text retrieval system** that learns a shared embedding space for images and natural language descriptions.  
Instead of predicting fixed class labels, the model retrieves and ranks images based on their semantic similarity to a given text query.

The system is trained on the **Flickr30k** dataset using **contrastive learning** and a **dual-encoder architecture**, inspired by modern multimodal retrieval models such as CLIP.

---

## Motivation
Traditional multimodal pipelines typically follow a classification-based approach:

- CNN for image feature extraction  
- RNN (LSTM / GRU) for text encoding  
- Feature fusion  
- Softmax-based classification  

While effective for fixed-label problems, this approach is not well suited for **retrieval tasks**, where the goal is to rank candidates dynamically.

Our task can be summarized as:

> *“Which images best match a given text description?”*

This motivates a **retrieval-based contrastive learning formulation**, where images and captions are embedded into a shared semantic space.

---

## Dataset
**Flickr30k** is used for training and evaluation.

- Contains images paired with multiple natural-language captions  
- Each training sample consists of a single image–caption pair  
- Other pairs in the same batch act as negative examples  

This dataset is widely used for multimodal learning and image–text retrieval tasks.

---

## System Architecture

The model follows a **dual-encoder design**:

### Image Encoder
- EfficientNet backbone  
- Pretrained on ImageNet  
- Classification head removed  
- Outputs a global visual feature vector  
- Fine-tuned during contrastive training  

### Text Encoder
- Vocabulary-based tokenizer  
- Token embeddings aggregated using **masked mean pooling**  
- Produces a fixed-length sentence representation  

Both encoders are followed by projection heads that map features into a shared embedding space.

---

## Projection Heads and Shared Embedding Space
- Image and text features are projected into a **256-dimensional space**
- Learnable linear layers are used
- L2 normalization places embeddings on the unit hypersphere

This enables stable similarity computation and efficient retrieval.

---

## Similarity and Scoring
- Similarity is computed using the **dot product**
- Due to normalization, this corresponds to **cosine similarity**
- A temperature-scaled softmax is used during training

---

## Training Methodology

### Contrastive Learning (InfoNCE)
- Training is performed using **InfoNCE contrastive loss**
- In each batch:
  - The correct image–caption pair is treated as a positive example
  - All other combinations serve as negatives
- The model is trained symmetrically for:
  - Image-to-text retrieval
  - Text-to-image retrieval

This can be interpreted as an implicit softmax-based classification over batch candidates.

---

## Optimization Details
- Optimizer: **AdamW**
- Learning rate: `2e-5`
- Batch size: `64`
- Epochs: `5`
- Train/validation split: `90% / 10%`
- Regularization:
  - Weight decay
  - Dropout inside encoder networks  

Full cross-validation was avoided due to computational cost.

---

## Quick start (real data, contrastive training)

Place the Flickr30k data here:  
`data/flickr30k/images/` and `data/flickr30k/annotations/annotations.csv`.

Install the packages:  
`pip install -r requirements.txt`.

Sanity check (loading, tokenization):  
`python main.py`.

Multimodal CLIP-style training (on image–caption pairs):  
`python train.py`.  
Saved to: `results/models/flickr30k_clip_best.pth` (model_state + vocab).

Save the image encoder after CLIP training:  
`python train_image_only.py` (saved to: `results/models/image_encoder_from_clip.pth`).
