"""Print the Gemini models this credential can actually use."""
from sim.client import GeminiClient, MissingCredential

try:
    models = GeminiClient.list_models()
except MissingCredential as error:
    raise SystemExit(str(error))

rows = [m for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
for model in sorted(rows, key=lambda m: m.get("name", "")):
    name = model.get("name", "").replace("models/", "")
    print(f"{name:46s} in={model.get('inputTokenLimit', '?'):>9} out={model.get('outputTokenLimit', '?'):>7}  {model.get('displayName','')}")
print(f"\n{len(rows)} usable models")
