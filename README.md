# Assignment 1 and Assignment 2

Student: Danabek Bosyn

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Assignment 1

```powershell
python main.py
```

## Assignment 2 notebook

Open `A2_Complete_Flow.ipynb` and run all cells.

## Assignment 2 training and evaluation

```powershell
python a2_src\train_sota.py --epochs 15
python a2_src\train_sota.py --ablation --epochs 15
python a2_src\evaluate_a2.py
```

## Live demo

```powershell
python -m streamlit run a2_app\app.py
```
