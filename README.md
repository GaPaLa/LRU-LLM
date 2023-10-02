# LRU-LLM
Implementing the LRU from the paper "Resurrecting Recurrent Neural Networks for Long Sequences" for language modelling.

simply:
  - wrapping the LRU layer code from the paper in a Flax module (after some edits for compatibility)
  - using the LRU layer within a transformer-style architecture, using it in place of the self-attention layer
  - adding character-level and llama-2 tokenizers
  - implementing autoregressive generation with nucleus sampling
  - adding a simple data loading and training pipeline
