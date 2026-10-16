.PHONY: help migrate api bot dashboard worker test lint clean

help:
	@echo "migrate   - create TimescaleDB hypertables + PostgreSQL tables"
	@echo "api       - run FastAPI control plane (port 8000)"
	@echo "bot       - run trading engine (paper mode by default)"
	@echo "dashboard - run Streamlit UI"
	@echo "worker    - run Celery worker (retrain/report)"
	@echo "test      - run pytest suite"

migrate:
	python scripts/migrate.py

api:
	python scripts/run_api.py

bot:
	python scripts/run_bot.py

dashboard:
	streamlit run dashboard/app.py

worker:
	celery -A wtb.workers.tasks worker --loglevel=INFO --concurrency=1

test:
	python -m pytest tests/ -q

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache
