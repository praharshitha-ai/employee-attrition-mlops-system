.PHONY: setup train api frontend test lint deploy clean

setup:
	pip install -r requirements.txt
	wandb login  # Manual after .env

train:
	python src/models/train.py

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	streamlit run frontend/app.py

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/
	black --check src/ tests/

deploy:
	docker-compose up --build

clean:
	rm -rf data/processed/ models/ experiments/

