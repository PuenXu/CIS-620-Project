import time
from main import main as run_sim
import importlib

def benchmark():
    maps = [("maps/map.json", 5), ("maps/map2.json", 9), ("maps/map3.json", 12)]
    results = []
    for map_path, n_robots in maps:
        for malicious in range(0, n_robots + 1):
            for use_learning in (False, True):
                import main
                importlib.reload(main)
                main.map = map_path
                main.num_malicious = malicious
                main.use_learning = use_learning
                main.animation = False
                main.record = False
                start = time.time()
                steps, completeness = main.main(max_steps=1200)
                elapsed = time.time() - start
                results.append({
                    "map": map_path,
                    "malicious": malicious,
                    "use_learning": use_learning,
                    "steps": steps,
                    "completeness": completeness,
                    "elapsed_sec": round(elapsed, 2),
                })
                print(results[-1])
    print("Done.")

if __name__ == "__main__":
    benchmark()
