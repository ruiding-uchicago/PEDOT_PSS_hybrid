"""Run a planned recipe through the black box; no corpus or identifiers."""
import json
from pathlib import Path
from pedot_pss import predict

experiment = json.loads(Path(__file__).with_name("polybot_recipe.json").read_text())
print(json.dumps(predict(experiment), indent=2))
