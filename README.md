# LRU-LLM
Implementing the LRU from the paper "Resurrecting Recurrent Neural Networks for Long Sequences" https://arxiv.org/abs/2303.06349 for use in language modelling.

simply:
  - wrapping the LRU layer code from the paper in a Flax module
  - using the LRU layer within a transformer-style architecture in place of self-attention
  - adding character-level and llama-2 tokenizers
  - implementing autoregressive generation with nucleus sampling
  - adding a simple data loading and training pipeline for TinyStories\
