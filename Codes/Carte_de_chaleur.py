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
    """
    Sert à l'affichage des différents graphs

    Args :\n
        size (tuple) : Taille de la fenetre graphique\n
        X, Y (Liste) : Données à tracer sur le graph\n
        S, C (Liste) : Données necessaires au différents tracé des cartes de chaleurs\n
        colorbar_label, title, xlabel, ylabel (str) : Titres des différentes parties du graph\n
        xlimit, ylimit (tuple) : Limites des axes du graph\n
        image, not_show : Paramètres servant uniquement de tests Vrai/Faux pour executer tel ou tel action
    """
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
    """
    Permet de récuper la frame en fonction d'un index

    Args :\n
        cap : Vidéo extraite par la bibliothèque CV2\n
        frame_index (int) : Indice de la frame à extraire

    Retruns :\n
        Frame : Renvoi la frame si l'index est reconnu\n
        None : Rien si CV2 ne trouve pas la frame
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        print(f"Erreur : frame non trouvée pour l'index {frame_index}.")
        return None


def afficher_carte_chaleur(
    instant, fixation_time_filtered, positions, width=None, height=None
):
    """
    Carte de chaleur basée sur le donées du CSV

    Args :\n
        instant (float) : instant auquel on veut tracer la carte de chaleur\n
        fixation_time_filtered (Liste) : Liste des temps pour le CSZ découpé\n
        positions (Liste) : Liste des Gaze pour le CSV découpé\n
        width, height (int) : Si données en arguments, définit l'échelle du graph, indépendemment de l'échelle de la vidéo
    """
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


def afficher_carte_chaleur_video(
    instant,
    video_path,
    media_frame_indices,
    media_timestamps,
    positions,
    fixation_time_filtered,
    tolerance=1e-6,
):
    """
    Carte de chaleur prenant en compte la vidéo et la frame en question

    Args :\n
        instant (float) : instant auquel on veut tracer la carte de chaleur\n
        video_path (str) : Chemin vers la vidéo\n
        media_frame_indices, media_timestamps (Liste) : Données du CSV complet\n
        positions, fixation_time_filtered (Liste) : Données du CSV découpé\n
        tolérance (float) : Tolérance avec laquelle on accepte les timecodes dans la liste (est plus une tolérance numérique et évité les bugs)
    """
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


def afficher_carte_chaleur_image(
    instant, fixation_time_filtered, positions, width, height
):
    """
    Carte de chaleur basée sur le donées du CSV en resteignant le graph

    Args :\n
        instant (float) : instant auquel on veut tracer la carte de chaleur\n
        fixation_time_filtered (Liste) : Liste des temps pour le CSZ découpé\n
        positions (Liste) : Liste des Gaze pour le CSV découpé\n
        width, height (int) : Si données en arguments, définit l'échelle du graph, indépendemment de l'échelle de la vidéo
    """
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
    afficher_carte_chaleur(instant, fixation_time_filtered, positions, width, height)
    afficher_carte_chaleur_video(
        instant,
        video_path,
        media_frame_indices,
        media_time_stamp,
        positions,
        fixation_time_filtered,
    )
    afficher_carte_chaleur_image(
        instant, fixation_time_filtered, positions, width, height
    )
    print("======================\n")


PROTECH_PATH = Path(__file__).resolve().parent.parent
