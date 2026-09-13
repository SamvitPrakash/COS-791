# COS-791
Onboarding:

First run in the prject root:
```bash
python -m venv .venv
# You only have to do this once. If you already have the .venv/ directory you can ignore this. 
```
Start the python virtual environemnt
```bash
source .venv/bin/activate
```
After any packages you install with pip you should run:
```bash
pip freeze > requirements.txt
```

To stop the virtual environment run:
```bash
deactivate
```

To run the files:
```bash
python -m src.main
```

Before you begin, your project should look something like this:
```bash
COS-791
├── data
├── .git
├── src
└── .venv
```

You should add your files into src/ and create a directory named after what you've implemented. Inside said directory you create an empty
```bash
__init__.py
```
file. You can paste your code here.

How to run:

1. Whole folder
```bash
python -m src.main --input-dir data/CHAOS_DATA --output-dir results/... --q 0.8
```
