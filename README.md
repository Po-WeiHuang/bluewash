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

7. 
6. Add in .env
 * OPENAI_API_KEY
 * REG_DIR: Regulation directory
 * CLAIMS_FILE: the output claim files after running the apo
 * REPORT_FILE

7. Execute main script
`python src/verdict_RF.py`