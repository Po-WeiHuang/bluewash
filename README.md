## Introduction

Detected Bluewashing claims based on UK, EU regulations and proofs from the paper.
Used all-MiniLM-L6-v2 for evidence retrieval with a Random Forest scoring engine. By applying a soft penalty to absolute marketing language and using SHAP logic to audit the model's internal decision-making. For every claim. OpenAI then converts these data points into a professional regulatory justification


## Running Steps

1. Installing uv

If you are on a Linux/macOS system (like your /home/huangp/ path suggests), run this in your terminal:

`curl -LsSf https://astral.sh/uv/install.sh | sh`

Restart your terminal after this so the uv command is recognized.

2. Adding information in setupenv.sh and 
`source setupenv.sh`


3. Creating a Virtual Environment with uv
`uv venv $UV_DIR`

4. sourcing the virtual environment
`source $UV_DIR/bin/acitivate`

5. Download dependencies
`uv pip install -r requirements.txt`

6. Add in .env
 * OPENAI_API_KEY: Add your openai api private key 
 * REG_DIR: Regulation directory (After Llamaparse) (default one is at llamainput/regulation)
 * CLAIMS_FILE: claim files (shall be acquired after finished 7.)
 * CLAIMS_SOURCE: the output claim files after running the repo (After Llamaparse) (default one is at llamainput/productinfo )
 * REPORT_FILE (Report Filename Path)

7. Make claims:
`python src/openaiapi.py`

8. Execute main script
`python src/verdict_RF.py`
