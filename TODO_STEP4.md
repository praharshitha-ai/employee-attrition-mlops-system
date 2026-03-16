# STEP 4 Hyperparameter Tuning - Detailed TODO

## Plan Approved: Optuna + W&B for RF tuning

1. [ ] Update src/models/train.py: Add Optuna objective, W&B integration
2. [ ] Create src/models/tune.py: Main tuning script (50 trials)
3. [ ] Install deps: pip install -r requirements.txt
4. [ ] wandb login (WANDB_API_KEY from .env)
5. [ ] Run python src/models/tune.py → best model logged
6. [ ] Update models/ with tuned model
7. [ ] Update TODO.md: STEP 4 ✅

*Next: STEP 5 W&B experiments*
