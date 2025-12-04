############################ Importation bibliothèques ############################
import numpy as np
import csv
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cv2
import sys
import os


############################ Fonction lecture et sauvegarde ############################
def lecture(file_name):
    datas = []
    with open(file_name, "r") as file:
        doc = csv.reader(file)
        for _ in range(6):
            next(doc)
        for line in doc:
            datas.append(line)
    return datas


def colonne_suite(file, column_name):
    index = 0
    for j in range(1, len(file[0])):
        if file[0][j] == column_name:
            index = j
    return [line[index] for line in file[1:]]


def insertion_valeurs(tableau, column, column_name):
    tableau.append([column_name])
    for elem in column:
        tableau[-1].append(elem)
    return tableau


def transpose(tableau, valeur_vide=""):
    max_len = max(len(line) for line in tableau)

    transposed_tableau = []
    for j in range(max_len):
        line = []
        for i in range(len(tableau)):
            if j < len(tableau[i]):
                line.append(tableau[i][j])
            else:
                line.append(valeur_vide)
        transposed_tableau.append(line)

    return transposed_tableau


def sauvegarde(file_name, lines):
    lines = transpose(lines)
    dossier = os.path.dirname(__file__)
    chemin_fichier = os.path.join(
        os.path.dirname(dossier) + "/outputs", file_name + ".csv"
    )

    with open(chemin_fichier, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(lines)

    print(f"Fichier CSV enregistré : {chemin_fichier}")


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
    image=None,
    not_show=None,
):
    plt.figure(figsize=size)
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
    if image:
        plt.imshow(frame, aspect="equal", extent=(0, width, 0, height))


def extraire_frame_par_index(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        print(f"Erreur : frame non trouvée pour l'index {frame_index}.")
        return None


def afficher_carte_chaleur_echelle(instants, width=None, height=None):
    for instant in instants:
        subset_positions = np.array(
            [
                pos
                for time, pos in zip(fixation_time_filtered, positions)
                if time <= instant
            ]
        )
        if len(subset_positions) > 0:
            display_graph(
                (8, 6),
                subset_positions[:, 0],
                subset_positions[:, 1],
                50,
                "red",
                None,
                f"Carte de chaleur - Instant: {instant} ms",
                "Gaze2dX(pixel)",
                "Gaze2dY(pixel)",
            )
            if width and height:
                display_graph(
                    (8, 6),
                    subset_positions[:, 0],
                    subset_positions[:, 1],
                    50,
                    "red",
                    None,
                    f"Carte de chaleur - Instant: {instant} ms",
                    "Gaze2dX(pixel)",
                    "Gaze2dY(pixel)",
                    (0, width),
                    (0, height),
                )
                display_graph(
                    (8, 6),
                    subset_positions[:, 0],
                    subset_positions[:, 1],
                    10,
                    "red",
                    None,
                    f"Carte de chaleur - Instant: {instant} ms",
                    "Gaze2dX(pixel)",
                    "Gaze2dY(pixel)",
                    (-width, 2 * width),
                    (-height, 2 * height),
                )


def afficher_carte_chaleur_avec_video(
    instants, video_path, media_frame_indices, media_timestamps, positions
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo.")
        return

    for instant in instants:
        if instant not in media_timestamps:
            print(f"Instant {instant} non trouvé dans les timestamps.")
            continue
        frame_index = media_frame_indices[media_timestamps.index(instant)]
        frame = extraire_frame_par_index(cap, frame_index)
        if frame is None:
            print(f"Frame non trouvée pour l'instant {instant}.")
            continue

        height, width, _ = frame.shape
        subset_positions = np.array(
            [
                [pos[0], height - pos[1]]
                for pos, time in zip(positions, fixation_time_filtered)
                if time <= instant
            ]
        )

        if len(subset_positions) > 0:
            display_graph(
                (8, 6),
                subset_positions[:, 0],
                subset_positions[:, 1],
                50,
                "red",
                None,
                f"Carte de chaleur - Instant: {instant:.2f} ms",
                "Gaze2dX(pixel)",
                "Gaze2dY(pixel)",
                (0, width),
                (0, height),
                1,
            )
            display_graph(
                (8, 6),
                subset_positions[:, 0],
                subset_positions[:, 1],
                10,
                "red",
                None,
                f"Carte de chaleur superposée - Instant: {instant:.2f} ms",
                "Gaze2dX(pixel)",
                "Gaze2dY(pixel)",
                (-width, 2 * width),
                (-height, 2 * height),
                1,
            )

    cap.release()


def afficher_carte_chaleur_echelle_image(instants, width, height):
    for instant in instants:
        subset_positions = np.array(
            [
                pos
                for time, pos in zip(fixation_time_filtered, positions)
                if time <= instant
            ]
        )
        if len(subset_positions) > 0:
            adjusted_positions = np.array(
                [[pos[0], height - pos[1]] for pos in subset_positions]
            )
            display_graph(
                (8, 6),
                adjusted_positions[:, 0],
                adjusted_positions[:, 1],
                50,
                "red",
                None,
                f"Carte de chaleur - Instant: {instant} ms (Échelle Image)",
                "Gaze2dX(pixel)",
                "Gaze2dY(pixel)",
                (0, width),
                (0, height),
            )
            display_graph(
                (8, 6),
                adjusted_positions[:, 0],
                adjusted_positions[:, 1],
                10,
                "red",
                None,
                f"Carte de chaleur - Instant: {instant} ms (Échelle Image)",
                "Gaze2dX(pixel)",
                "Gaze2dY(pixel)",
                (-width, 2 * width),
                (-height, 2 * height),
            )


def afficher_carte_chaleur_avec_derniere_frame_valide(
    tfinal_stamp, video_path, media_frame_indices, media_time_stamp, positions
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo.")
        return

    frame_index = media_frame_indices[media_time_stamp.index(tfinal_stamp)]
    last_frame = None

    while last_frame is None and frame_index >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_index -= 1

    if last_frame is None:
        print("Aucune frame valide trouvée pour l'instant final.")
        cap.release()
        return

    height, width, _ = last_frame.shape
    subset_positions = np.array(
        [
            [pos[0], height - pos[1]]
            for pos, time in zip(positions, fixation_time_filtered)
            if time <= tfinal_stamp
        ]
    )

    if len(subset_positions) > 0:
        display_graph(
            (8, 6),
            subset_positions[:, 0],
            subset_positions[:, 1],
            50,
            "red",
            None,
            f"Carte de chaleur superposée - Dernière frame valide à l'instant final",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            (0, width),
            (0, height),
            1,
        )
        display_graph(
            (8, 6),
            subset_positions[:, 0],
            subset_positions[:, 1],
            10,
            "red",
            None,
            f"Carte de chaleur superposée - Dernière frame valide à l'instant final",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            (-width, 2 * width),
            (-height, 2 * height),
            1,
        )

    cap.release()


def afficher_carte_chaleur_avec_video_pour_instant(
    instant,
    video_path,
    media_frame_indices,
    media_timestamps,
    positions,
    tolerance=1e-6,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo.")
        return

    index = next(
        (
            i
            for i, ts in enumerate(media_timestamps)
            if np.isclose(ts, instant, atol=tolerance)
        ),
        None,
    )
    if index is None:
        print(f"Instant {instant} non trouvé dans les timestamps.")
        return

    frame_index = media_frame_indices[index]
    frame = extraire_frame_par_index(cap, frame_index)
    if frame is None:
        print(f"Frame non trouvée pour l'instant {instant}.")
        return

    height, width, _ = frame.shape
    subset_positions = np.array(
        [
            [pos[0], height - pos[1]]
            for pos, time in zip(positions, fixation_time_filtered)
            if time <= instant
        ]
    )

    if len(subset_positions) > 0:
        display_graph(
            (8, 6),
            subset_positions[:, 0],
            subset_positions[:, 1],
            50,
            "red",
            None,
            f"Carte de chaleur superposée - Instant: {instant:.6f} ms",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            (0, width),
            (0, height),
            1,
        )
        display_graph(
            (8, 6),
            subset_positions[:, 0],
            subset_positions[:, 1],
            10,
            "red",
            None,
            f"Carte de chaleur superposée - Instant: {instant:.6f} ms",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            (-width, 2 * width),
            (-height, 2 * height),
            1,
        )

    cap.release()


def extraire_donnees_utiles_excel(
    file_name, instants, media_frame_indices, media_time_stamp, positions
):
    feuille = []

    insertion_valeurs(feuille, instants, "Instants (ms)")
    frame_indices = [
        (
            media_frame_indices[media_time_stamp.index(instant)]
            if instant in media_time_stamp
            else None
        )
        for instant in instants
    ]
    insertion_valeurs(feuille, frame_indices, "Frame Indices")
    insertion_valeurs(feuille, media_time_stamp, "Media TimeStamp (ms)")
    pos_x = [pos[0] for pos in positions]
    pos_y = [pos[1] for pos in positions]
    insertion_valeurs(feuille, pos_x, "Gaze2dX (pixel)")
    insertion_valeurs(feuille, pos_y, "Gaze2dY (pixel)")

    sauvegarde(file_name, feuille)
    print(
        f"Les données utiles ont été extraites et sauvegardées dans le fichier {file_name}.xlsx"
    )


def exporter_donnees_clusters_avec_ordre(
    pos_x, pos_y, counts, fixation_order, filename="Clusters_Fixation_Ordre"
):
    sheet = [
        ["Gaze2dX (pixel)"],
        ["Gaze2dY (pixel)"],
        ["counts"],
        ["ordre des clusters"],
    ]
    for i in range(len(pos_x)):
        sheet[0].append(pos_x[i])
        sheet[1].append(pos_y[i])
        sheet[2].append(counts[i])
        sheet[3].append(fixation_order[i])
    sauvegarde(filename, sheet)
    print(f"Données avec ordre exportées avec succès dans le fichier {filename}.")


def display_graph_arrow(
    size,
    X,
    Y,
    S,
    C,
    colorbar_label,
    title,
    xlabel,
    ylabel,
    hwidth,
    hlength,
    xlimit=None,
    ylimit=None,
):
    display_graph(
        size, X, Y, S, C, colorbar_label, title, xlabel, ylabel, not_show=True
    )
    for idx, (x, y) in enumerate(zip(X, Y)):
        plt.text(
            x, y, str(idx + 1), fontsize=12, ha="center", va="center", color="black"
        )
    previous_position = None
    for i, (current_x, current_y) in enumerate(zip(X, Y)):
        if previous_position is not None:
            plt.arrow(
                previous_position[0],
                previous_position[1],
                current_x - previous_position[0],
                current_y - previous_position[1],
                head_width=hwidth,
                head_length=hlength,
                fc="blue",
                ec="blue",
                length_includes_head=True,
            )
        previous_position = (current_x, current_y)
    plt.gca().set_aspect("equal", adjustable="box")
    if xlimit and ylimit:
        plt.xlim(xlimit)
        plt.ylim(ylimit)
    plt.show()


def clusters_fixations(size, X, Y, C, title1, title2, width=None, height=None):
    display_graph_arrow(
        size,
        X,
        Y,
        800,
        C,
        "Nombre de Fixations",
        title1,
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        0.005,
        0.01,
    )
    display_graph_arrow(
        size,
        X,
        Y,
        800,
        C,
        "Nombre de Fixations",
        title1 + " : echelle video",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        0.005,
        0.01,
        (0, width),
        (0, height),
    )
    display_graph_arrow(
        size,
        X,
        Y,
        200,
        C,
        "Nombre de Fixations",
        title1 + " : echelle etendue",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        0.005,
        0.01,
        (-width, 2 * width),
        (-height, 2 * height),
    )

    if last_frame is not None:
        adjusted_pos_y = height - np.array(Y)

        display_graph_arrow(
            size,
            X,
            adjusted_pos_y,
            800,
            C,
            "Nombre de Fixations",
            title2,
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            10,
            15,
            (0, width),
            (0, height),
        )
        display_graph_arrow(
            size,
            X,
            adjusted_pos_y,
            200,
            C,
            "Nombre de Fixations",
            title2,
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            10,
            15,
            (-width, 2 * width),
            (-height, 2 * height),
        )
        display_graph_arrow(
            size,
            X,
            adjusted_pos_y,
            800,
            C,
            "Nombre de Fixations",
            title1 + " (Échelle Image)",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            0.005,
            0.01,
            (0, width),
            (0, height),
        )
        display_graph_arrow(
            size,
            X,
            adjusted_pos_y,
            200,
            C,
            "Nombre de Fixations",
            title1 + " (Échelle Étendue)",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            0.005,
            0.01,
            (-width, 2 * width),
            (-height, 2 * height),
        )

    else:
        print("Impossible d'extraire la dernière frame de la vidéo.")


############################ lecture fichier ###########
""" Fichier 21s : 20241015_0001_00.mp4 et 20241015_0001_00.csv
    Fichier 3m32s : 20250325_0001_00.mp4 et 20250325_0001_00.csv"""

fichier = lecture(sys.argv[1])
video_path = sys.argv[2]
############################ lecture variables ###########
gaze2dx = colonne_suite(fichier, "Gaze2dX")
gaze2dy = colonne_suite(fichier, "Gaze2dY")
gaze_class = colonne_suite(fichier, "GazeClass")
blink = colonne_suite(fichier, "Blink")
MediaTimeStamp = colonne_suite(fichier, "MediaTimeStamp")
gaze_right_reciprocal_distance = colonne_suite(fichier, "GazeRightReciprocalDistance")
gaze_left_reciprocal_distance = colonne_suite(fichier, "GazeLeftReciprocalDistance")
confidence = colonne_suite(fichier, "Confidence")
vergence = colonne_suite(fichier, "VergenceAngle")
media_frame_indices = colonne_suite(fichier, "MediaFrameIndex")
############################ Nettoyage variables de gaze ###########
fixations_idx = [i for i, gclass in enumerate(gaze_class) if gclass == "F"]
fixation_gaze2dx = [gaze2dx[i] for i in fixations_idx]
fixation_gaze2dy = [gaze2dy[i] for i in fixations_idx]
fixation_gaze2dx_filtered = []
fixation_gaze2dy_filtered = []
for x, y in zip(fixation_gaze2dx, fixation_gaze2dy):
    try:
        fixation_gaze2dx_filtered.append(float(x))
        fixation_gaze2dy_filtered.append(float(y))
    except ValueError:
        pass
positions = np.array(list(zip(fixation_gaze2dx_filtered, fixation_gaze2dy_filtered)))
############################ Algorithme DBSCAN pour le clustering ############################
eps_optimal = 100
db = DBSCAN(eps=eps_optimal, min_samples=2).fit(positions)
labels = db.labels_
cluster_counts = Counter(labels[labels != -1])
unique_labels = set(labels)
clusters = {}
for label in unique_labels:
    if label != -1:
        cluster_positions = positions[labels == label]
        avg_position = np.mean(cluster_positions, axis=0)
        clusters[tuple(avg_position)] = cluster_counts[label]
############################ Calcul temps de fixation cumulé ############################
fixation_time_stamp = [MediaTimeStamp[i] for i in fixations_idx]
fixation_time_filtered = []
for t in fixation_time_stamp:
    try:
        fixation_time_filtered.append(float(t))
    except ValueError:
        pass
index_label_mapping = {
    fixations_idx[i]: labels[i] for i in range(len(fixations_idx)) if i < len(labels)
}
fixation_times = []
for i in range(len(fixation_time_filtered) - 1):
    current_idx = fixations_idx[i]
    next_idx = fixations_idx[i + 1]
    if current_idx in index_label_mapping and next_idx in index_label_mapping:
        current_cluster = index_label_mapping[current_idx]
        next_cluster = index_label_mapping[next_idx]
        if current_cluster == next_cluster:
            fixation_time = fixation_time_filtered[i + 1] - fixation_time_filtered[i]
            fixation_times.append(fixation_time)
        else:
            fixation_times.append(0)
cluster_fixation_time = {}
for label in set(labels):
    if label != -1:
        cluster_fixation_time[label] = sum(
            fixation_times[i]
            for i, fix_idx in enumerate(fixations_idx[:-1])
            if fix_idx in index_label_mapping and index_label_mapping[fix_idx] == label
        )
############################ Visualisation échelle des données ############################
if len(clusters) > 0:
    pos_x, pos_y = zip(*clusters.keys())
    counts = list(clusters.values())

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
    )
else:
    print("Aucun cluster n'a été trouvé.")
    sys.exit()
############################ Visualisation sans frame vidéo ############################
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    sys.exit()
else:
    last_frame_index = int(media_frame_indices[-1])
    last_frame = None

    while last_frame is None and last_frame_index >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        ret, frame = cap.read()
        if ret:
            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            last_frame_index -= 1
cap.release()

### Echelle vidéo
height, width, _ = last_frame.shape

display_graph(
    (8, 6),
    pos_x,
    pos_y,
    800,
    counts,
    "Nombre de Fixations",
    "Clusters de Fixation (échelle video)",
    "Gaze2dX(pixel)",
    "Gaze2dY(pixel)",
    (0, width),
    (0, height),
)
display_graph(
    (8, 6),
    pos_x,
    pos_y,
    200,
    counts,
    "Nombre de Fixations",
    "Clusters de Fixation (echelle etendue)",
    "Gaze2dX(pixel)",
    "Gaze2dY(pixel)",
    (-1 * width, 2 * width),
    (-1 * height, 2 * height),
)
############################ Visualisation avec frame vidéo ############################

### Echelle vidéo

if last_frame is not None:
    height, width, _ = last_frame.shape
    inverted_frame = np.flipud(last_frame)

    adjusted_pos_y = height - np.array(pos_y)
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        800,
        counts,
        "Nombre de Fixations",
        "Superposition de la Dernière Frame avec les Clusters",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (0, width),
        (0, height),
        1,
    )
else:
    print("Impossible d'extraire la dernière frame de la vidéo.")

### Echelle étendue

if last_frame is not None:
    height, width, _ = last_frame.shape
    inverted_frame = np.flipud(last_frame)

    adjusted_pos_y = height - np.array(pos_y)
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        200,
        counts,
        "Nombre de Fixations",
        "Superposition de la Dernière Frame avec les Clusters",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (-1 * width, 2 * width),
        (-1 * height, 2 * height),
        1,
    )
else:
    print("Impossible d'extraire la dernière frame de la vidéo.")

###

if last_frame is not None:
    adjusted_pos_y = height - np.array(pos_y)
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        800,
        counts,
        "Nombre de Fixations",
        "Clusters de Fixation ajustés à l'échelle de l'image",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (0, width),
        (0, height),
    )
else:
    print("Impossible d'extraire les dimensions de l'image.")

###

if last_frame is not None:
    adjusted_pos_y = height - np.array(pos_y)
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        200,
        counts,
        "Nombre de Fixations",
        "Clusters de Fixation ajustés à l'échelle de l'image",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (-width, 2 * width),
        (-height, 2 * height),
    )
else:
    print("Impossible d'extraire les dimensions de l'image.")

### Extraction des données

if len(clusters) > 0:
    pos_x, pos_y = zip(*clusters.keys())
    counts = list(clusters.values())

    feuille = []

    insertion_valeurs(feuille, pos_x, "Gaze2dX (pixel)")
    insertion_valeurs(feuille, pos_y, "Gaze2dY (pixel)")
    insertion_valeurs(feuille, counts, "Nombre de Fixations")

    image_width = [width] * len(pos_x)
    image_height = [height] * len(pos_y)
    insertion_valeurs(feuille, image_width, "Image Width (pixel)")
    insertion_valeurs(feuille, image_height, "Image Height (pixel)")

    sauvegarde("Clusters_Fixation", feuille)
else:
    print("Aucun cluster à exporter.")
############################ Temps de fixation par cluster ############################

### Echelle des données

if len(cluster_fixation_time) > 0:
    pos_x, pos_y = zip(*clusters.keys())
    times = list(cluster_fixation_time.values())

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
    )
    display_graph(
        (8, 6),
        pos_x,
        pos_y,
        800,
        times,
        "Temps de fixation cumulé (ms)",
        "Temps de Fixation cumulé par Cluster échelle video",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (0, width),
        (0, height),
    )
    display_graph(
        (8, 6),
        pos_x,
        pos_y,
        200,
        times,
        "Temps de fixation cumulé (ms)",
        "Temps de Fixation cumulé par Cluster échelle etendue",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (-1 * width, 2 * width),
        (-1 * height, 2 * height),
    )
else:
    print("Aucun cluster n'a été trouvé.")
    sys.exit()

###

if last_frame is not None:
    height, width, _ = last_frame.shape
    adjusted_pos_y = height - np.array(pos_y)

    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        800,
        times,
        "Temps de fixation cumulé (ms)",
        "Superposition de la Dernière Frame avec les Clusters",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (0, width),
        (0, height),
        1,
    )
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        200,
        times,
        "Temps de fixation cumulé (ms)",
        "Superposition de la Dernière Frame avec les Clusters",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (-1 * width, 2 * width),
        (-height, 2 * height),
        1,
    )
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        800,
        times,
        "Temps de fixation cumulé (ms)",
        "Temps de Fixation cumulé par Cluster (échelle image)",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (0, width),
        (0, height),
    )
    display_graph(
        (10, 8),
        pos_x,
        adjusted_pos_y,
        200,
        times,
        "Temps de fixation cumulé (ms)",
        "Temps de Fixation cumulé par Cluster (échelle image)",
        "Gaze2dX(pixel)",
        "Gaze2dY(pixel)",
        (-width, 2 * width),
        (-height, 2 * height),
    )
else:
    print("Impossible d'extraire la dernière frame de la vidéo.")

### Extravation des données

if len(clusters) > 0 and len(cluster_fixation_time) > 0:
    pos_x, pos_y = zip(*clusters.keys())
    times = list(cluster_fixation_time.values())
    counts = list(clusters.values())

    feuille = []

    insertion_valeurs(feuille, pos_x, "Gaze2dX (pixel)")
    insertion_valeurs(feuille, pos_y, "Gaze2dY (pixel)")
    insertion_valeurs(feuille, counts, "Nombre de Fixations")
    insertion_valeurs(feuille, times, "Temps de Fixation cumulé (ms)")

    image_width = [width] * len(pos_x)
    image_height = [height] * len(pos_y)
    insertion_valeurs(feuille, image_width, "Image Width (pixel)")
    insertion_valeurs(feuille, image_height, "Image Height (pixel)")

    sauvegarde("Clusters_Temps_Fixation", feuille)
else:
    print("Aucun cluster ou temps de fixation cumulé à exporter.")
############################ Carte de Chaleur (t0) ############################

### Echelle données

media_frame_indices = [
    int(idx) for idx in media_frame_indices if idx not in [None, "NA"]
]
media_time_stamp = [float(ts) for ts in MediaTimeStamp if ts not in [None, "NA"]]
positions = np.array(
    [
        [float(gaze2dx[i]), float(gaze2dy[i])]
        for i in fixations_idx
        if gaze2dx[i] not in [None, "NA"] and gaze2dy[i] not in [None, "NA"]
    ]
)

### Def des temps

t0 = fixation_time_filtered[0]
tmilieu = fixation_time_filtered[len(fixation_time_filtered) // 2]
tfinal = fixation_time_filtered[-1]

### Affichage graphs

afficher_carte_chaleur_echelle([t0], width, height)
afficher_carte_chaleur_avec_video(
    [t0], video_path, media_frame_indices, media_time_stamp, positions
)
afficher_carte_chaleur_echelle_image([t0], width, height)
############################ Carte de Chaleur (tmilieu) ############################
afficher_carte_chaleur_echelle([tmilieu], width, height)
afficher_carte_chaleur_avec_video(
    [tmilieu], video_path, media_frame_indices, media_time_stamp, positions
)
afficher_carte_chaleur_echelle_image([tmilieu], width, height)
############################ Carte de Chaleur (tfinal) ############################
afficher_carte_chaleur_echelle([tfinal], width, height)
afficher_carte_chaleur_avec_video(
    [tfinal], video_path, media_frame_indices, media_time_stamp, positions
)
afficher_carte_chaleur_echelle_image([tfinal], width, height)
############################ Recherche dernière frame ############################

#!! Si le dernier frame n'existe pas, on essaye de chercher les frames d'avant. Une fois trouvé, on la superpose avec la carte de chaleur finale

afficher_carte_chaleur_avec_derniere_frame_valide(
    tfinal, video_path, media_frame_indices, media_time_stamp, positions
)
############################ Carte de chaleur instant définie ############################
random_instant = 4.335055784

afficher_carte_chaleur_echelle([random_instant], width, height)
afficher_carte_chaleur_avec_video_pour_instant(
    random_instant, video_path, media_frame_indices, media_time_stamp, positions
)
afficher_carte_chaleur_echelle_image([random_instant], width, height)

### Extraction des données

"""Instants (ms) :
Liste des instants (t_0), (t_{milieu}),(t_{random}) (t_{final}).

Indices des frames (Frame Indices) :
Indices des frames correspondant aux instants donnés.

Timestamps (Media TimeStamp (ms)) :
Liste complète des timestamps disponibles dans les données.

Positions des fixations (Gaze2dX (pixel) et Gaze2dY (pixel)) :
Positions des fixations ((x), (y)) utilisées pour générer les cartes de chaleur."""

extraire_donnees_utiles_excel(
    "Donnees_cartes_de_chaleur",
    [t0, tmilieu, random_instant, tfinal],
    media_frame_indices,
    media_time_stamp,
    positions,
)
############################ Cluster fixations avec couleur proportionnelles et flèches ############################
clusters_fixations(
    (10, 8),
    pos_x,
    pos_y,
    counts,
    "Clusters de Fixation et Flèches pour le Chemin",
    "Superposition de la Dernière Frame avec les Clusters et Flèches pour le Chemin",
    width,
    height,
)

### Extraction des données
"""
pos_x: Liste des coordonnées X des clusters.
pos_y: Liste des coordonnées Y des clusters.
counts: Liste des cardinalités des clusters.
fixation_order: Ordre temporel des clusters. """

fixation_order = list(range(1, len(pos_x) + 1))
exporter_donnees_clusters_avec_ordre(pos_x, pos_y, counts, fixation_order)
############################ Clusters de Fixation (> seuil positions) avec Flèches pour le Chemin (Taille Constante) ############################
seuil = 20
filtered_clusters = {k: v for k, v in clusters.items() if v > seuil}
filtered_pos_x, filtered_pos_y = zip(*filtered_clusters.keys())
filtered_counts = list(filtered_clusters.values())

clusters_fixations(
    (10, 8),
    filtered_pos_x,
    filtered_pos_y,
    filtered_counts,
    f"Clusters de Fixation (> {seuil} positions) avec Flèches pour le Chemin",
    f"Superposition de la Dernière Frame avec les Clusters (> {seuil} positions)",
    width,
    height,
)
############################ Clusters de Fixation avec Couleur Proportionnelle et Flèches pour le Chemin (Taille Constante) en choisissant top_n plus grands clusters ############################
top_n = 3
top_clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)[:top_n]
top_pos_x, top_pos_y = zip(*[k for k, v in top_clusters])
top_counts = [v for k, v in top_clusters]

clusters_fixations(
    (10, 8),
    top_pos_x,
    top_pos_y,
    top_counts,
    f"Top {top_n} Clusters de Fixation avec Flèches",
    f"Superposition de la Dernière Frame avec les Top {top_n} Clusters",
    width,
    height,
)
