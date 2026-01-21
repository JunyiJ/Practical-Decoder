# Practical-Decoder
A repo to implement the decoder-only GPT-style model and play around with different attention mechanism and MoE structures


## To train the model
`python -m src.train.train`

## To generate with the trained checkpoint
`python -m src.utils.generate --checkpoint checkpoints/final.pt --data-path data/raw/shakespeare.txt --prompt "To be, or not to be" --device mps`
