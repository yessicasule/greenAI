# Training Configs

QAT training hyperparameters currently live inline as constants in
`training/scripts/kaggle_qat_trainer.py`'s `CELL 1` block (`MODEL_NAME`,
`DATA_PATH`, learning rate, epochs, LoRA rank/alpha, bit-width cycle).

This directory is the intended home for extracting those into a real
`qat_config.yaml` so hyperparameters have one source of truth instead of
being edited in-place inside a notebook-style script. Deferred as a
follow-up — rewiring the script to read from a config file is a real code
change to a script currently being relied on for training, and shouldn't
happen the same day it's about to be run for real.

The runtime inference config (bit-width tiers, fuzzy controller parameters)
is a separate thing — that's `backend/src/green_weight/config.yaml`, loaded
by the app itself, not a training hyperparameter file.
