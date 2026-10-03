from pathlib import Path
for p in ["/content/e005/src", "/content/e005/data", "/content/e005/runtime", "/content/e005/outputs", "/content/e005/model"]:
    Path(p).mkdir(parents=True, exist_ok=True)
print("remote directories ready")
