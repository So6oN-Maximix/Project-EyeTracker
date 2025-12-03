from unittest.mock import patch
from Codes.Console_execution import verifier_fichier_video


def test_video_valide_path_mock():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        resultat = verifier_fichier_video("Datas/20241015_0001_00.mp4", "vidéo")
        assert resultat == "Valide"


def test_video_introuvable_path_mock():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False
        resultat = verifier_fichier_video("video_fantome.mp4", "video")
        assert resultat == "Introuvable"


def test_mauvais_format_path():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        resultat = verifier_fichier_video("mon_cv.txt", "csv")
        assert resultat == "Format Incorrect - Demandé CSV"
