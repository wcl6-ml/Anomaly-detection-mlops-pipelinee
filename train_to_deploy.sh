#!/bin/bash
python src/prepare_data.py
python src/training.py
python src/utils/register_model.py
python src/utils/export_model.py
docker-compose up -d --build
