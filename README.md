# Installation
conda env create -f environment.yml && conda activate wikicheck

or

pip install -r requirements.txt


# Backend

uvicorn main:app --reload