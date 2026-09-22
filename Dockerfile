# CPU-only, runs the test suite. Nothing here renders, so no GL is installed.
#   docker build -t so-arm100-rl . && docker run --rm so-arm100-rl
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY so_arm100_rl ./so_arm100_rl
COPY tests ./tests
COPY train.py eval_gap.py compare_gap.py export_onnx.py verify.sh ./
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "pytest", "tests"]
