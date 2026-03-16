# Detailed TODO from Approved Plan (STEPS 2-3)

## 1. STEP 2 - Data Layer (Neon PostgreSQL)
- [ ] Get full Neon DB connection details (host, port, user, password, dbname from billowing-meadow-16884461 project)
- [ ] Update .env with NEON_URL
- [ ] Run `python data/scripts/upload_to_neon.py` or equivalent to create table/load CSV data
- [ ] Verify data loaded (query count == 1470 rows)

## 2. STEP 3 - Model Training & Pipeline
- [x] Implement src/models/pipeline.py
- [x] Implement src/models/train.py
- [ ] Update src/data/preprocessing.py for integration
- [ ] pip install -r requirements.txt (if joblib/wandb added)
- [ ] Run `python src/data/eda.py` (if plots missing)
- [ ] Run `python src/models/train.py`
- [ ] Update TODO.md progress

## Next: User confirm DB creds → Execute steps iteratively

*Updated by BLACKBOXAI*
