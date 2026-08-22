# Training Configs

**Correction 2026-08-22:** this README used to point at
`training/scripts/kaggle_qat_trainer.py` as the source of QAT
hyperparameters. That was wrong — diffing that script's `LoraConfig`
against the real trained adapters' `adapter_config.json` shows they don't
match (wrong rank/alpha per tier, wrong `target_modules`, wrong adapter
naming). `kaggle_qat_trainer.py` (and its notebook counterpart
`major-project-v2.ipynb`) are a stale, unused pair that were never
archived to `_legacy/` — see `CLAUDE.md`'s "QAT Training" section and
`NEW.md` Phase 3 for the full finding. **The real training script is
`training/scripts/adapter-training.ipynb`**, and it's a Kaggle notebook,
not a `.py` file — externalizing its inline hyperparameters into a
`qat_config.yaml` would mean rewiring notebook cells, not a script.

Given the adapters are now confirmed loadable and functional (real load
test, 2026-08-22), retraining is only needed if Session 2 shows them
underperforming plain quantization — so this externalization work stays
deferred, but for a different reason now: it's simply not urgent until a
retrain is actually needed, not because touching the script is risky
(nothing here is "about to be run for real" today).

The runtime inference config (bit-width tiers, fuzzy controller parameters)
is a separate thing — that's `backend/src/green_weight/config.yaml`, loaded
by the app itself, not a training hyperparameter file.
