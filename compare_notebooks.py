#!/usr/bin/env python3
# compare_notebooks_robust.py
# Wymagania: nbformat, pandas
# Użycie:
#   python compare_notebooks_robust.py --nb1 Scoring.ipynb --nb2 pipeline.ipynb --sample_csv test_subset.csv
# Jeśli --sample_csv nie podane, skrypt spróbuje uruchomić ale pominie komórki wczytujące brakujące pliki.

import nbformat, argparse, ast, io, hashlib, json, traceback, contextlib, os
import pandas as pd

def read_notebook(path):
    return nbformat.read(path, as_version=4)

def detect_io_cells(nb):
    """Statyczna analiza AST: zwraca listę indeksów komórek, które zawierają odczyt pliku lub transformacje DF."""
    cells = [c for c in nb.cells if c.cell_type == "code"]
    interesting = []
    for idx, cell in enumerate(cells):
        src = cell.source
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = ast.unparse(node.func) if hasattr(ast, 'unparse') else ''
                if any(k in cname for k in ("read_csv", "read_parquet", "read_excel", "to_csv", "DataFrame", "merge", "concat", "groupby", "pivot", "melt", "assign", "pipe", "transform", "fit_transform")):
                    interesting.append(idx)
                    break
    return sorted(set(interesting))

def execute_cells_with_snapshots(nb, sample_csv_path=None, allow_io_rewrite=True):
    """Wykonuje komórki z notatnika; jeśli napotka pd.read_csv(...) i podano sample_csv_path,
       podmienia ścieżkę na sample_csv_path aby operować na podzbiorze testowym."""
    cells = [c for c in nb.cells if c.cell_type == "code"]
    env = {"pd": pd, "__name__":"__main__"}
    snapshots = []
    for idx, cell in enumerate(cells):
        code = cell.source
        exec_code = code
        if sample_csv_path and "read_csv" in code and allow_io_rewrite:
            # prosta podmiana: replace first occurrence of pd.read_csv('...') or read_csv("...") etc.
            # Uwaga: to proste podejście — powinno działać dla zwykłych przypadków.
            exec_code = exec_code.replace("pd.read_csv(", f"pd.read_csv(r'{sample_csv_path}', ")
            exec_code = exec_code.replace("read_csv(", f"pd.read_csv(r'{sample_csv_path}', ")
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(exec_code, env)
        except Exception:
            # zapisz błąd, ale kontynuuj
            snapshots.append({"cell_index": idx, "error": traceback.format_exc()})
            continue
        # po wykonaniu komórki zbieramy wszystkie obiekty typu DataFrame
        for name, obj in list(env.items()):
            if isinstance(obj, pd.DataFrame):
                try:
                    df = obj
                    cols = list(df.columns)
                    dtypes = {c: str(df[c].dtype) for c in cols}
                    row_count = len(df)
                    sample = df if row_count <= 1000 else df.head(1000)
                    csv_b = sample.to_csv(index=False).encode('utf-8')
                    md5 = hashlib.md5(csv_b).hexdigest()
                    # small preview
                    preview = sample.head(5).to_dict(orient='list')
                    snapshots.append({
                        "cell_index": idx,
                        "var_name": name,
                        "columns": cols,
                        "dtypes": dtypes,
                        "row_count": row_count,
                        "md5": md5,
                        "preview": preview,
                    })
                except Exception as e:
                    snapshots.append({"cell_index": idx, "var_name": name, "error": str(e)})
    return snapshots

def columns_jaccard(a,b):
    a=set(a); b=set(b)
    if not a and not b: return 1.0
    return len(a & b)/len(a | b)

def snapshot_similarity(a,b):
    cols_sim = columns_jaccard(a.get("columns",[]), b.get("columns",[]))
    row_sim = 1.0 if a.get("row_count") == b.get("row_count") else 0.0
    md5_sim = 1.0 if a.get("md5") == b.get("md5") else 0.0
    score = 0.6*cols_sim + 0.2*row_sim + 0.2*md5_sim
    return {"score":score,"cols_sim":cols_sim,"row_sim":row_sim,"md5_sim":md5_sim}

def compare_snapshots(snaps1, snaps2):
    matches = []
    for i,s in enumerate(snaps1):
        best = {"score":-1}
        for j,t in enumerate(snaps2):
            sim = snapshot_similarity(s,t)
            if sim["score"]>best["score"]:
                best = {"score":sim["score"], "j":j, "sim":sim}
        matches.append({"i":i, "cell1": s.get("cell_index"), "var1": s.get("var_name"), 
                        "j": best.get("j"), "cell2": (snaps2[best["j"]].get("cell_index") if best.get("j") is not None else None),
                        "var2": (snaps2[best["j"]].get("var_name") if best.get("j") is not None else None),
                        "score":best["score"], "sim":best.get("sim")})
    return matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb1", required=True)
    parser.add_argument("--nb2", required=True)
    parser.add_argument("--sample_csv", required=False, help="ścieżka do pliku CSV z testowym podzbiorem (opcjonalnie)")
    args = parser.parse_args()

    nb1 = read_notebook(args.nb1)
    nb2 = read_notebook(args.nb2)

    print("Statyczna analiza (gdzie są IO/transformacje)...")
    import ast
    def detect(nb):
        cells = [c for c in nb.cells if c.cell_type=="code"]
        ints=[]
        for idx,cell in enumerate(cells):
            try:
                tree = ast.parse(cell.source)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = ast.unparse(node.func) if hasattr(ast,'unparse') else ""
                    if any(k in fn for k in ("read_csv","read_parquet","DataFrame","merge","concat","groupby","assign","transform","fit_transform","pipeline")):
                        ints.append(idx); break
        return sorted(set(ints))
    print("nb1 interesting cells:", detect(nb1))
    print("nb2 interesting cells:", detect(nb2))

    print("\nWykonuję notebook 1 (próbka: sample_csv={})".format(args.sample_csv))
    snaps1 = execute_cells_with_snapshots(nb1, sample_csv_path=args.sample_csv)
    print("Znaleziono {} snapshotów w nb1".format(len(snaps1)))
    print("Wykonuję notebook 2")
    snaps2 = execute_cells_with_snapshots(nb2, sample_csv_path=args.sample_csv)
    print("Znaleziono {} snapshotów w nb2".format(len(snaps2)))

    matches = compare_snapshots(snaps1, snaps2)
    out = {"snaps1":snaps1,"snaps2":snaps2,"matches":matches}
    outpath = "compare_steps_report.json"
    with open(outpath,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print("Zapisano raport do", outpath)

if __name__ == "__main__":
    main()
