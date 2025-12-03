import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path


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
        plt.imshow(image[0], aspect="equal", extent=(0, image[1], 0, image[2]))
    if not not_show:
        file_name = title + ".png"
        plt.savefig(PROTECH_PATH / "Outputs" / "Graphs" / file_name)
        print(f"Image enregistrée : {PROTECH_PATH/'Outputs'/'Graphs'/file_name}")
    plt.close()


def extraire_frame_par_index(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        print(f"Erreur : frame non trouvée pour l'index {frame_index}.")
        return None


def afficher_carte_chaleur_echelle(
    instant, fixation_time_filtered, positions, width=None, height=None
):
    subset_positions = np.array(
        [pos for time, pos in zip(fixation_time_filtered, positions) if time <= instant]
    )
    if len(subset_positions) > 0:
        display_graph(
            (8, 6),
            subset_positions[:, 0],
            subset_positions[:, 1],
            50,
            "red",
            None,
            f"Carte de chaleur - Instant {instant:.2f} s",
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
                f"Carte de chaleur - Instant {instant:.2f} s - échelle video",
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
                f"Carte de chaleur - Instant {instant:.2f} s - échelle étendue",
                "Gaze2dX(pixel)",
                "Gaze2dY(pixel)",
                (-width, 2 * width),
                (-height, 2 * height),
            )


def afficher_carte_chaleur_echelle_video(
    instant,
    video_path,
    media_frame_indices,
    media_timestamps,
    positions,
    fixation_time_filtered,
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
        cap.release()
        return

    frame_index = media_frame_indices[index]
    frame_initial = media_frame_indices[0]
    frame = extraire_frame_par_index(cap, frame_index - frame_initial)

    if frame is None:
        print(f"Frame non trouvée pour l'instant {instant} (index {frame_index}).")
        cap.release()
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
            f"Carte de chaleur - Instant {instant:.2f} s - Échelle vidéo",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            (0, width),
            (0, height),
            (frame, width, height),
        )
        display_graph(
            (8, 6),
            subset_positions[:, 0],
            subset_positions[:, 1],
            10,
            "red",
            None,
            f"Carte de chaleur - Instant {instant:.2f} s - Échelle étendue",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            (-width, 2 * width),
            (-height, 2 * height),
            (frame, width, height),
        )

    cap.release()


def afficher_carte_chaleur_echelle_image(
    instant, fixation_time_filtered, positions, width, height
):
    subset_positions = np.array(
        [pos for time, pos in zip(fixation_time_filtered, positions) if time <= instant]
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
            f"Carte de chaleur - Instant {instant:.2f} s  - échelle video",
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
            f"Carte de chaleur - Instant {instant:.2f} s  - échelle étendue",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            (-width, 2 * width),
            (-height, 2 * height),
        )


def afficher_carte_chaleur_avec_derniere_frame_valide(
    tfinal_stamp,
    video_path,
    media_frame_indices,
    media_time_stamp,
    positions,
    fixation_time_filtered,
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
            f"Carte de chaleur superposée - Dernière frame valide à l'instant final - échelle video",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            (0, width),
            (0, height),
            (frame, width, height),
        )
        display_graph(
            (8, 6),
            subset_positions[:, 0],
            subset_positions[:, 1],
            10,
            "red",
            None,
            f"Carte de chaleur superposée - Dernière frame valide à l'instant final - échelle étendue",
            "Gaze2dX(pixel)",
            "Gaze2dY(pixel)",
            (-width, 2 * width),
            (-height, 2 * height),
            (frame, width, height),
        )

    cap.release()


def display_cartes_chaleur(
    instant,
    fixation_time_filtered,
    positions,
    width,
    height,
    video_path,
    media_frame_indices,
    media_time_stamp,
):
    print(f"=== INSTANT {instant:.2f} s ===")
    afficher_carte_chaleur_echelle(
        instant, fixation_time_filtered, positions, width, height
    )
    afficher_carte_chaleur_echelle_video(
        instant,
        video_path,
        media_frame_indices,
        media_time_stamp,
        positions,
        fixation_time_filtered,
    )
    afficher_carte_chaleur_echelle_image(
        instant, fixation_time_filtered, positions, width, height
    )
    print("======================\n")


PROTECH_PATH = Path(__file__).resolve().parent.parent
