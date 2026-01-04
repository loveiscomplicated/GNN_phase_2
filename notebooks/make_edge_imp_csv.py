import csv
import re


import os
cur_dir = os.path.dirname(__file__)
par_dir = os.path.join(cur_dir, '..')

ad_txt_dir = os.path.join('resources', 'PGExplainer_edge_importance_admission.txt')
ad_save_dir = os.path.join('resources', 'PGExplainer_edge_importance_admission.csv')

dis_txt_dir = os.path.join('resources', 'PGExplainer_edge_importance_discharge.txt')
dis_save_dir = os.path.join('resources', 'PGExplainer_edge_importance_discharge.csv')


def txt_to_csv(input_path, output_path):
    pattern = re.compile(r"(\w+)\s*->\s*(\w+):\s*([0-9.]+)")

    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                source, target, score = match.groups()
                rows.append([source, target, float(score)])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "score"])
        writer.writerows(rows)

    print(f"{len(rows)} rows saved to {output_path}")

if __name__ == "__main__":
    txt_to_csv(ad_txt_dir, ad_save_dir)
    txt_to_csv(dis_txt_dir, dis_save_dir)

