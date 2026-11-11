import sys
from pathlib import Path

# Aggiungi cartella al path
_ROOT = Path.cwd().resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import run_challenge_grid, resolve_project_root, save_challenge_artifacts
import yaml

PROJECT_ROOT = resolve_project_root(_ROOT)
MODEL_CONFIG = "hypergraph/whnn"

LIFTINGS = ["khop", "knn", "kernel", "modularity_maximization"]

def main():
    lifting_yaml_path = PROJECT_ROOT / "configs" / "transforms" / "liftings" / "graph2hypergraph_default.yaml"

    for lifting in LIFTINGS:
        print(f"\n======================================")
        print(f"🚀 INIZIO VALUTAZIONE CON LIFTING: {lifting}")
        print(f"======================================\n")

        # Sovrascriviamo fisicamente il file di default per evitare problemi con la sintassi complessa di Hydra
        config_content = f"""defaults:
  - /transforms/liftings/graph2hypergraph@graph2hypergraph_lifting: {lifting}
"""
        with open(lifting_yaml_path, "w") as f:
            f.write(config_content)

        # Eseguiamo la grid (limitata a 2 run per Kaggle test, o completa se vogliamo)
        # Togliamo limit_runs per la versione finale
        results, study_id = run_challenge_grid(
            project_root=PROJECT_ROOT,
            model_config=MODEL_CONFIG,
            study_id=f"whnn_lifting_{lifting}",
            quiet=False,
        )

        # Salviamo i risultati separatamente per ogni lifting
        output_paths = save_challenge_artifacts(
            results,
            model_config=MODEL_CONFIG,
            study_id=f"whnn_{lifting}",
        )
        print(f"✅ Risultati per {lifting} salvati in {output_paths['dir']}")

if __name__ == "__main__":
    main()
