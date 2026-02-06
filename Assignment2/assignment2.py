import os
import numpy as np
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "players_stats_by_season_full_details.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Element-wise division; returns 0 where denom is 0."""
    numer = numer.astype(float, copy=False)
    denom = denom.astype(float, copy=False)
    out = np.zeros_like(numer, dtype=float)
    mask = denom != 0
    out[mask] = numer[mask] / denom[mask]
    return out


def normalize_name(s: str) -> str:
    #Normalizing the column names for a flexible matching.
    return s.strip().lower().replace(" ", "_")


def find_col(dtype_names, *candidates: str) -> str:
    
    # Finding a column name in dtype_names given possible candidates and handling NumPy genfromtxt renaming (for instance from '3PM' to 'f3PM').
    names = list(dtype_names)
    norm_map = {normalize_name(n): n for n in names}

    # Trying direct normalized match, plus common NumPy prefix behavior for the invlid identifiers
    for cand in candidates:
        c = normalize_name(cand)

        # the exact normalized match
        if c in norm_map:
            return norm_map[c]

        # numpy sometimes prefixes 'f' for invalid field names (like starting with a digit)
        if ("f" + c) in norm_map:
            return norm_map["f" + c]

        # sometimes it may keep digits but replace punctuation; try a few variants
        # (best-effort; won’t crash your program for minor header differences)
        alt = c.replace("%", "pct")
        if alt in norm_map:
            return norm_map[alt]
        if ("f" + alt) in norm_map:
            return norm_map["f" + alt]

    # If not found, show available names to help you fix quickly
    print("\nAvailable columns in CSV:")
    print(", ".join(names))
    raise KeyError(f"Missing required column. Tried candidates: {candidates}")


    # I used a class to keep the data loading, metric calculations, and exporting organized. 
    # It helped avoid repeated code and made the workflow easier to follow. 
class NBASeasonAnalyzer:
    #Here is the Loading of the NBA season CSV into a NumPy structured ndarray and computes per-row (player-season) metrics

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = self._load_csv()

        # Resolving the key columns we need 
        self.col_season = find_col(self.data.dtype.names, "Season")
        self.col_player = find_col(self.data.dtype.names, "Player")
        self.col_gp = find_col(self.data.dtype.names, "GP")
        self.col_min = find_col(self.data.dtype.names, "MIN")

        self.col_fgm = find_col(self.data.dtype.names, "FGM")
        self.col_fga = find_col(self.data.dtype.names, "FGA")
        self.col_3pm = find_col(self.data.dtype.names, "3PM", "FG3M", "3PM_per_game")
        self.col_3pa = find_col(self.data.dtype.names, "3PA", "FG3A", "3PA_per_game")
        self.col_ftm = find_col(self.data.dtype.names, "FTM")
        self.col_fta = find_col(self.data.dtype.names, "FTA")

        self.col_pts = find_col(self.data.dtype.names, "PTS")
        self.col_blk = find_col(self.data.dtype.names, "BLK")
        self.col_stl = find_col(self.data.dtype.names, "STL")

    def _load_csv(self) -> np.ndarray:
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)  # we handle the commas inside quotes correctly
            header = next(reader)

            rows = [row for row in reader if row]  # (skipping empty lines)

    # Building a structured array of strings
        dtype = [(h.strip(), "U200") for h in header]
        data = np.zeros(len(rows), dtype=dtype)

        for i, row in enumerate(rows):
        # If a row is shorter or longer ---> pading and trimming safely
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[: len(header)]

            for j, h in enumerate(header):
                data[h.strip()][i] = row[j]

        return data


    def to_float(arr):
        s = np.char.strip(arr.astype(str))
        s = np.where(s == "", "0", s)
        x = s.astype(float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)



    def compute_metrics(self) -> np.ndarray:
        season = self.data[self.col_season].astype(str)
        player = self.data[self.col_player].astype(str)

        gp = self.data[self.col_gp].astype(float)
        mins = self.data[self.col_min].astype(float)

        fgm = self.data[self.col_fgm].astype(float)
        fga = self.data[self.col_fga].astype(float)
        pm3 = self.data[self.col_3pm].astype(float)
        pa3 = self.data[self.col_3pa].astype(float)
        ftm = self.data[self.col_ftm].astype(float)
        fta = self.data[self.col_fta].astype(float)

        pts = self.data[self.col_pts].astype(float)
        blk = self.data[self.col_blk].astype(float)
        stl = self.data[self.col_stl].astype(float)

        #Required accuracies
        fg_acc = safe_divide(fgm, fga)
        tp_acc = safe_divide(pm3, pa3)
        ft_acc = safe_divide(ftm, fta)

        #Points per minute
        pts_per_min = safe_divide(pts, mins)

        #Points per game
        pts_per_game = safe_divide(pts, gp)

        #Overall shooting accuracy
        #(FGM + FTM) / (FGA + FTA)
        overall_acc = safe_divide(fgm + ftm, fga + fta)

        # Blocks/steals per each game
        blk_per_game = safe_divide(blk, gp)
        stl_per_game = safe_divide(stl, gp)

        out_dtype = [
            ("player", "U100"),
            ("season", "U20"),
            ("gp", float),
            ("min", float),
            ("fg_acc", float),
            ("tp_acc", float),
            ("ft_acc", float),
            ("pts_per_min", float),
            ("pts_per_game", float),
            ("overall_acc", float),
            ("blk_per_game", float),
            ("stl_per_game", float),
        ]

        out = np.zeros(player.size, dtype=out_dtype)
        out["player"] = player
        out["season"] = season
        out["gp"] = gp
        out["min"] = mins
        out["fg_acc"] = fg_acc
        out["tp_acc"] = tp_acc
        out["ft_acc"] = ft_acc
        out["pts_per_min"] = pts_per_min
        out["pts_per_game"] = pts_per_game
        out["overall_acc"] = overall_acc
        out["blk_per_game"] = blk_per_game
        out["stl_per_game"] = stl_per_game

        return out

    @staticmethod   
    #I used static methods for helper functions that don’t rely on instance data like exporting arrays to files. 
    #They behave like the normal functions but keeprelated logic grouped within the class.
    def top_n(metrics: np.ndarray, field: str, n: int = 100) -> np.ndarray:
        # Sort ascending, take last n, reverse for descending
        order = np.argsort(metrics[field])
        order = order[::-1]
        return metrics[order][: min(n, metrics.size)]

    @staticmethod
    def save_tsv(path: str, table: np.ndarray):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = "\t".join(table.dtype.names)

        # Build formats: strings for player/season, floats for everything else
        fmts = []
        for name in table.dtype.names:
            if table.dtype[name].kind in ("U", "S"):
                fmts.append("%s")
            else:
                fmts.append("%.6f")

        np.savetxt(path, table, delimiter="\t", fmt=fmts, header=header, comments="")

    def export_all(self, metrics: np.ndarray, out_dir: str = OUTPUT_DIR):
        os.makedirs(out_dir, exist_ok=True)

        # full table
        self.save_tsv(os.path.join(out_dir, "metrics_by_player_season.tsv"), metrics)

        #the top 100 for required metrics
        tops = {
            "top100_fg_accuracy.tsv": "fg_acc",
            "top100_3pt_accuracy.tsv": "tp_acc",
            "top100_ft_accuracy.tsv": "ft_acc",
            "top100_points_per_game.tsv": "pts_per_game",
            "top100_overall_accuracy.tsv": "overall_acc",
            "top100_blocks_per_game.tsv": "blk_per_game",
            "top100_steals_per_game.tsv": "stl_per_game",
        }

        for fname, field in tops.items():
            top = self.top_n(metrics, field, n=100)

            # Save only player, season, metric
            small_dtype = [("player", "U100"), ("season", "U20"), (field, float)]
            small = np.zeros(top.size, dtype=small_dtype)
            small["player"] = top["player"]
            small["season"] = top["season"]
            small[field] = top[field]

            self.save_tsv(os.path.join(out_dir, fname), small)

        print(f"Saved outputs to: {out_dir}")
        print(f" Total player-season rows analyzed: {metrics.size}")


def main():
    analyzer = NBASeasonAnalyzer(CSV_PATH)
    metrics = analyzer.compute_metrics()
    analyzer.export_all(metrics, out_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
