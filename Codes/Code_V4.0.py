############################ Importation bibliothèques ############################
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cv2
import sys

# On importe les différents programmes necessaires à l'exécution
import Reduction_video
import Console_execution
import Test_timeline
import Carte_de_chaleur

############################ Fonctions Utilitaires Optimisées ############################


def get_output_path(filename):
    """
    Permet de récupérer le chemin de sorite pour enregistrer les graphs\n

    Args :\n
        filename (str) : Nom du fichier à enregister

    Returns :\n
        str : Chemin du fichier que l'on veut enregister
    """
    return Reduction_video.PROTECH_PATH / "Outputs" / "Graphs" / filename


def display_graph(
    size,
    X,
    Y,
    S,
    C,
    colorbar_label,
    title,
    xlabel,
    ylabel,
    xlimit=None,
    ylimit=None,
    bg_image=None,
    save_only=False,
    arrow=False,
    arrow_params=None,
):
    """ """
    plt.figure(figsize=size)

    if bg_image is not None:
        h, w = bg_image.shape[:2]
        plt.imshow(bg_image, aspect="equal", extent=(0, w, 0, h))

    plt.scatter(X, Y, s=S, c=C, cmap="YlOrRd", alpha=0.8)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if xlimit and ylimit:
        plt.xlim(xlimit)
        plt.ylim(ylimit)
        plt.gca().set_aspect("equal", adjustable="box")

    if arrow:
        for idx, (x, y) in enumerate(zip(X, Y)):
            plt.text(
                x,
                y,
                str(idx + 1),
                fontsize=12,
                ha="center",
                va="center",
                color="black",
                weight="bold",
            )

        if len(X) > 1:
            for i in range(len(X) - 1):
                plt.arrow(
                    X[i],
                    Y[i],
                    X[i + 1] - X[i],
                    Y[i + 1] - Y[i],
                    head_width=arrow_params.get("hw", 10),
                    head_length=arrow_params.get("hl", 15),
                    fc="blue",
                    ec="blue",
                    length_includes_head=True,
                )

    file_name = title + ".png"
    save_path = get_output_path(file_name)
    plt.savefig(save_path)
    print(f"Image enregistrée : {save_path}")
    plt.close()


def extraire_donnees_utiles_excel(file_name, instants_cibles, df_full):
    output_data = {
        "Instants (ms)": instants_cibles,
        "Frame Indices": [],
        "Media TimeStamp (ms)": [],
        "Gaze2dX (pixel)": [],
        "Gaze2dY (pixel)": [],
    }

    for t in instants_cibles:
        row = df_full[df_full["MediaTimeStamp"] == t]
        if not row.empty:
            output_data["Frame Indices"].append(row.iloc[0]["MediaFrameIndex"])
            output_data["Media TimeStamp (ms)"].append(t)
            output_data["Gaze2dX (pixel)"].append(row.iloc[0]["Gaze2dX"])
            output_data["Gaze2dY (pixel)"].append(row.iloc[0]["Gaze2dY"])
        else:
            output_data["Frame Indices"].append(None)
            output_data["Media TimeStamp (ms)"].append(None)
            output_data["Gaze2dX (pixel)"].append(None)
            output_data["Gaze2dY (pixel)"].append(None)

    df_out = pd.DataFrame(output_data)
    df_out_T = df_out.T
    Reduction_video.sauvegarde(file_name, df_out_T.reset_index().values.tolist())
    print(f"Fichier CSV enregistré {file_name}")


def save_dataframe_as_transposed_csv(filename, data_dict):
    df = pd.DataFrame(data_dict)
    df_T = df.T
    Reduction_video.sauvegarde(filename, df_T.reset_index().values.tolist())
    print(f"Fichier CSV enregistré : {filename}")


def clusters_fixations_plot_suite(
    X, Y, C, counts, title_base, title_superpo, width, height, last_frame
):
    size_std = (8, 6)
    size_large = (10, 8)

    # Graphs simples (fond blanc)
    display_graph(
        size_std,
        X,
        Y,
        800,
        C,
        "Nombre de Fixations",
        title_base,
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        arrow=True,
        arrow_params={"hw": 0.005, "hl": 0.01},
        save_only=True,
    )
    display_graph(
        size_std,
        X,
        Y,
        800,
        C,
        "Nombre de Fixations",
        title_base + " - échelle video",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        xlimit=(0, width),
        ylimit=(0, height),
        arrow=True,
        arrow_params={"hw": 0.005, "hl": 0.01},
        save_only=True,
    )
    display_graph(
        size_std,
        X,
        Y,
        200,
        C,
        "Nombre de Fixations",
        title_base + " - échelle étendue",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        xlimit=(-width, 2 * width),
        ylimit=(-height, 2 * height),
        arrow=True,
        arrow_params={"hw": 0.005, "hl": 0.01},
        save_only=True,
    )

    # Graphs avec superposition vidéo
    if last_frame is not None:
        adjusted_Y = height - np.array(Y)

        display_graph(
            size_large,
            X,
            adjusted_Y,
            800,
            C,
            "Nombre de Fixations",
            title_superpo,
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            xlimit=(0, width),
            ylimit=(0, height),
            bg_image=np.flipud(last_frame),
            arrow=True,
            arrow_params={"hw": 10, "hl": 15},
            save_only=True,
        )
        display_graph(
            size_large,
            X,
            adjusted_Y,
            200,
            C,
            "Nombre de Fixations",
            title_superpo + " - échelle étendue",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            xlimit=(-width, 2 * width),
            ylimit=(-height, 2 * height),
            bg_image=np.flipud(last_frame),
            arrow=True,
            arrow_params={"hw": 10, "hl": 15},
            save_only=True,
        )
        display_graph(
            size_large,
            X,
            adjusted_Y,
            800,
            C,
            "Nombre de Fixations",
            title_base + " - échelle video (Ajusté)",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            xlimit=(0, width),
            ylimit=(0, height),
            arrow=True,
            arrow_params={"hw": 0.005, "hl": 0.01},
            save_only=True,
        )
        display_graph(
            size_large,
            X,
            adjusted_Y,
            200,
            C,
            "Nombre de Fixations",
            title_base + " - échelle étendue (Ajusté)",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            xlimit=(-width, 2 * width),
            ylimit=(-height, 2 * height),
            arrow=True,
            arrow_params={"hw": 0.005, "hl": 0.01},
            save_only=True,
        )


############################ Execution Principale ############################
video_name, csv_file, input_timecodes, csv_file_timeline = Console_execution.main()
Test_timeline.creation_timeline(csv_file_timeline)

new_csv_path, new_video_name = Reduction_video.reduction(
    video_name, csv_file, input_timecodes[0], input_timecodes[1]
)
full_video_path = Reduction_video.PROTECH_PATH / "Outputs" / new_video_name
csv_path = Reduction_video.PROTECH_PATH / "Outputs" / new_csv_path

df = pd.read_csv(csv_path, sep=",")
df.columns = df.columns.str.strip()

cols_numeric = [
    "Gaze2dX",
    "Gaze2dY",
    "MediaTimeStamp",
    "GazeRightReciprocalDistance",
    "GazeLeftReciprocalDistance",
    "Confidence",
    "VergenceAngle",
    "MediaFrameIndex",
]
for col in cols_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("============ CLUSTERS DE FIXATION ============")
df_fix = df[
    (df["GazeClass"] == "F") & df["Gaze2dX"].notna() & df["Gaze2dY"].notna()
].copy()

positions = df_fix[["Gaze2dX", "Gaze2dY"]].values

if len(positions) == 0:
    print("Aucune fixation valide trouvée.")
    sys.exit()

eps_optimal = 100
db = DBSCAN(eps=eps_optimal, min_samples=2).fit(positions)
df_fix["Cluster"] = db.labels_

df_clusters = df_fix[df_fix["Cluster"] != -1].copy()

if df_clusters.empty:
    print("Aucun cluster n'a été trouvé.")
    sys.exit()

cluster_summary = (
    df_clusters.groupby("Cluster")
    .agg(
        Centroid_X=("Gaze2dX", "mean"),
        Centroid_Y=("Gaze2dY", "mean"),
        Count=("Cluster", "count"),
    )
    .reset_index()
)

pos_x = cluster_summary["Centroid_X"].tolist()
pos_y = cluster_summary["Centroid_Y"].tolist()
counts = cluster_summary["Count"].tolist()
labels_uniques = cluster_summary["Cluster"].tolist()

df_fix["Next_TimeStamp"] = df_fix["MediaTimeStamp"].shift(-1)
df_fix["Next_Cluster"] = df_fix["Cluster"].shift(-1)
df_fix["Duration"] = df_fix["Next_TimeStamp"] - df_fix["MediaTimeStamp"]
df_fix["Valid_Duration"] = np.where(
    df_fix["Cluster"] == df_fix["Next_Cluster"], df_fix["Duration"], 0
)

times_series = (
    df_fix[df_fix["Cluster"] != -1].groupby("Cluster")["Valid_Duration"].sum()
)
times = [times_series.get(lbl, 0) for lbl in labels_uniques]

display_graph(
    (8, 6),
    pos_x,
    pos_y,
    800,
    counts,
    "Nombre de Fixations",
    "Clusters de Fixation",
    "Gaze2dX(pixel)",
    "Gaze2dY(pixel)",
    save_only=True,
)

print("Lecture vidéo pour extraction frame...")
cap = cv2.VideoCapture(str(full_video_path))

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    sys.exit()

last_frame_idx = int(df["MediaFrameIndex"].max())
last_frame = None

cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
ret, frame = cap.read()
if ret:
    last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
else:
    for i in range(10):  # Essayer les 10 dernières frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx - i)
        ret, frame = cap.read()
        if ret:
            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            break
cap.release()

if last_frame is None:
    print("Impossible d'extraire la dernière frame.")
    height, width = 1080, 1920  # Valeurs par défaut au cas où
else:
    height, width, _ = last_frame.shape

display_graph(
    (8, 6),
    pos_x,
    pos_y,
    800,
    counts,
    "Nombre de Fixations",
    "Clusters de Fixation  - échelle video",
    "Gaze2dX(pixel)",
    "Gaze2dY(pixel)",
    xlimit=(0, width),
    ylimit=(0, height),
    save_only=True,
)
display_graph(
    (8, 6),
    pos_x,
    pos_y,
    200,
    counts,
    "Nombre de Fixations",
    "Clusters de Fixation  - échelle étendue",
    "Gaze2dX(pixel)",
    "Gaze2dY(pixel)",
    xlimit=(-width, 2 * width),
    ylimit=(-height, 2 * height),
    save_only=True,
)

if last_frame is not None:
    adjusted_pos_y = height - np.array(pos_y)

    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        800,
        counts,
        "Nombre de Fixations",
        "Superposition de la Dernière Frame avec les Clusters - echelle video",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        xlimit=(0, width),
        ylimit=(0, height),
        bg_image=np.flipud(last_frame),
        save_only=True,
    )
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        200,
        counts,
        "Nombre de Fixations",
        "Superposition de la Dernière Frame avec les Clusters - echelle étendue",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        xlimit=(-width, 2 * width),
        ylimit=(-height, 2 * height),
        bg_image=np.flipud(last_frame),
        save_only=True,
    )

    # Graphs ajustés sans background explicite mais coordonnées inversées
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        800,
        counts,
        "Nombre de Fixations",
        "Clusters de Fixation ajustés à l'échelle de l'image - echelle video",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        xlimit=(0, width),
        ylimit=(0, height),
        save_only=True,
    )
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        200,
        counts,
        "Nombre de Fixations",
        "Clusters de Fixation ajustés à l'échelle de l'image - echelle étendue",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        xlimit=(-width, 2 * width),
        ylimit=(-height, 2 * height),
        save_only=True,
    )

save_dataframe_as_transposed_csv(
    "Clusters_Fixation.csv",
    {
        "Gaze2dX (pixel)": pos_x,
        "Gaze2dY (pixel)": pos_y,
        "Nombre de Fixations": counts,
        "Image Width (pixel)": [width] * len(pos_x),
        "Image Height (pixel)": [height] * len(pos_y),
    },
)

if len(times) > 0:
    display_graph(
        (8, 6),
        pos_x,
        pos_y,
        800,
        times,
        "Temps de fixation cumulé (ms)",
        "Temps de Fixation cumulé par Cluster",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        save_only=True,
    )
    display_graph(
        (8, 6),
        pos_x,
        pos_y,
        800,
        times,
        "Temps de fixation cumulé (ms)",
        "Temps de Fixation cumulé par Cluster - échelle video",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        xlimit=(0, width),
        ylimit=(0, height),
        save_only=True,
    )
    display_graph(
        (8, 6),
        pos_x,
        pos_y,
        200,
        times,
        "Temps de fixation cumulé (ms)",
        "Temps de Fixation cumulé par Cluster - échelle etendue",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        xlimit=(-width, 2 * width),
        ylimit=(-height, 2 * height),
        save_only=True,
    )

    if last_frame is not None:
        adjusted_pos_y = height - np.array(pos_y)
        display_graph(
            (10, 8),
            pos_x,
            adjusted_pos_y,
            800,
            times,
            "Temps de fixation cumulé (ms)",
            "Superposition de la Dernière Frame avec les Clusters 01",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            xlimit=(0, width),
            ylimit=(0, height),
            bg_image=np.flipud(last_frame),
            save_only=True,
        )
        display_graph(
            (10, 8),
            pos_x,
            adjusted_pos_y,
            200,
            times,
            "Temps de fixation cumulé (ms)",
            "Superposition de la Dernière Frame avec les Clusters 02",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            xlimit=(-width, 2 * width),
            ylimit=(-height, 2 * height),
            bg_image=np.flipud(last_frame),
            save_only=True,
        )
        display_graph(
            (10, 8),
            pos_x,
            adjusted_pos_y,
            800,
            times,
            "Temps de fixation cumulé (ms)",
            "Temps de Fixation cumulé par Cluster - échelle video 01",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            xlimit=(0, width),
            ylimit=(0, height),
            save_only=True,
        )
        display_graph(
            (10, 8),
            pos_x,
            adjusted_pos_y,
            200,
            times,
            "Temps de fixation cumulé (ms)",
            "Temps de Fixation cumulé par Cluster - échelle video 01",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            xlimit=(-width, 2 * width),
            ylimit=(-height, 2 * height),
            save_only=True,
        )

    save_dataframe_as_transposed_csv(
        "Clusters_Temps_Fixation.csv",
        {
            "Gaze2dX (pixel)": pos_x,
            "Gaze2dY (pixel)": pos_y,
            "Nombre de Fixations": counts,
            "Temps de Fixation cumulé (ms)": times,
            "Image Width (pixel)": [width] * len(pos_x),
            "Image Height (pixel)": [height] * len(pos_y),
        },
    )
else:
    print("Aucun temps de fixation cumulé à exporter.")

print("========== FIN CLUSTERS DE FIXATION ==========\n")
############################ Carte de Chaleur ############################
print("============== CARTE DE CHALEUR ==============")

heat_positions = df_fix[["Gaze2dX", "Gaze2dY"]].values
heat_timestamps = df_fix["MediaTimeStamp"].tolist()
full_frame_indices = df["MediaFrameIndex"].tolist()
full_timestamps = df["MediaTimeStamp"].tolist()

if len(heat_timestamps) >= 3:
    t0 = heat_timestamps[0]
    tmilieu = heat_timestamps[len(heat_timestamps) // 2]
    tfinal = heat_timestamps[-1]

    Carte_de_chaleur.display_cartes_chaleur(
        t0,
        heat_timestamps,
        heat_positions,
        width,
        height,
        full_video_path,
        full_frame_indices,
        full_timestamps,
    )
    Carte_de_chaleur.display_cartes_chaleur(
        tmilieu,
        heat_timestamps,
        heat_positions,
        width,
        height,
        full_video_path,
        full_frame_indices,
        full_timestamps,
    )
    Carte_de_chaleur.display_cartes_chaleur(
        tfinal,
        heat_timestamps,
        heat_positions,
        width,
        height,
        full_video_path,
        full_frame_indices,
        full_timestamps,
    )

    extraire_donnees_utiles_excel(
        "Donnees_cartes_de_chaleur.csv", [t0, tmilieu, tfinal], df
    )

print("============ FIN CARTE DE CHALEUR ============\n")
############################ Clusters avec Flèches ############################
print("======== CLUSTERS DE FIXATION FLECHES ========")

clusters_fixations_plot_suite(
    pos_x,
    pos_y,
    counts,
    counts,
    "Clusters de Fixation et Flèches pour le Chemin",
    "Superposition de la Dernière Frame avec les Clusters et Flèches pour le Chemin",
    width,
    height,
    last_frame,
)

fixation_order = list(range(1, len(pos_x) + 1))
save_dataframe_as_transposed_csv(
    "Clusters_Fixation_Ordre.csv",
    {
        "Gaze2dX (pixel)": pos_x,
        "Gaze2dY (pixel)": pos_y,
        "counts": counts,
        "ordre des clusters": fixation_order,
    },
)

seuil = 20
df_clusters_thresh = cluster_summary[cluster_summary["Count"] > seuil]
if not df_clusters_thresh.empty:
    cl_x = df_clusters_thresh["Centroid_X"].tolist()
    cl_y = df_clusters_thresh["Centroid_Y"].tolist()
    cl_counts = df_clusters_thresh["Count"].tolist()

    clusters_fixations_plot_suite(
        cl_x,
        cl_y,
        cl_counts,
        cl_counts,
        f"Clusters de Fixation (sup {seuil} positions) avec Flèches pour le Chemin",
        f"Superposition de la Dernière Frame avec les Clusters (sup {seuil} positions)",
        width,
        height,
        last_frame,
    )

top_n = 3
df_clusters_top = cluster_summary.nlargest(top_n, "Count")
if not df_clusters_top.empty:
    cl_x = df_clusters_top["Centroid_X"].tolist()
    cl_y = df_clusters_top["Centroid_Y"].tolist()
    cl_counts = df_clusters_top["Count"].tolist()

    clusters_fixations_plot_suite(
        cl_x,
        cl_y,
        cl_counts,
        cl_counts,
        f"Top {top_n} Clusters de Fixation avec Flèches",
        f"Superposition de la Dernière Frame avec les Top {top_n} Clusters",
        width,
        height,
        last_frame,
    )

print("====== FIN CLUSTERS DE FIXATION FLECHES ======\n")
print("Tous les fichiers ont été enregistrés !!")
print(
    "================================= FIN DE L'ANALYSE ================================="
)
