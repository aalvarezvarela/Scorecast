#!/usr/bin/env python3
"""Localiza resúmenes NBA de Motion Station y, opcionalmente, los descarga.

Edita las variables del bloque «Configuración» y ejecuta el archivo o sus celdas
en orden (marcadas con # %%). No necesita argumentos de línea de comandos.

DOWNLOAD_MODE: audio (solo pista de audio de YouTube), video, both (vídeo y
audio extraído del vídeo) o local_audio (extraer de LOCAL_VIDEO sin consultar
ESPN ni YouTube). AUDIO_FORMAT puede ser mp3, m4a o wav.

Requiere: pip install requests yt-dlp; ffmpeg y ffprobe para vídeo/audio.
Descarga únicamente si tienes los permisos necesarios.
"""

# %% Importaciones
import csv
import re
import shutil
import subprocess
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

# %% Configuración
DATE = "2026-10-28"  # Fecha del partido en EE. UU. (AAAA-MM-DD)
# Localizar la raíz tanto al ejecutar el archivo como las celdas interactivas.
PROJECT_ROOT = next(
    parent
    for parent in Path(globals().get("__file__", Path.cwd() / "get_game_recaps.py"))
    .resolve()
    .parents
    if (parent / "pyproject.toml").is_file()
)
OUTPUT = PROJECT_ROOT / "data" / "nba_highlights"  # Carpeta de destino
LIST_ONLY = False  # True: solo localiza enlaces; no descarga ni extrae archivos
DOWNLOAD_MODE = "audio"  # "audio", "video", "both" o "local_audio"
AUDIO_FORMAT = "mp3"  # "mp3", "m4a" o "wav"
LOCAL_VIDEO = None  # En modo local_audio: ruta al vídeo, p. ej. "resumen.mp4"
SEARCH_RESULTS = 25  # Resultados de búsqueda por partido (>= 1)
DIAGNOSTICS = True  # Mostrar resultados y motivos por los que se descartan
CHANNEL_ID = "UCLd4dSmXdrJykO_hgOzbfPw"  # Motion Station (@motionstation4342)

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
CSV_COLUMNS = [
    "date",
    "espn_game_id",
    "away",
    "home",
    "title",
    "url",
    "status",
    "video_path",
    "audio_path",
]


# %% Funciones auxiliares
def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def matches_video_date(info: dict, game_date: date) -> bool:
    """Motion Station publica el resumen el día del partido o al siguiente.

    Los títulos no incluyen fecha. Usar la publicación en horario del este de
    EE. UU.; si solo hay upload_date, YouTube la expresa en UTC.
    """
    timestamp = info.get("release_timestamp") or info.get("timestamp")
    if timestamp is not None:
        published_date = datetime.fromtimestamp(
            timestamp, ZoneInfo("America/New_York")
        ).date()
    else:
        upload_date = info.get("upload_date")
        if not upload_date:
            return False
        try:
            published_date = datetime.strptime(upload_date, "%Y%m%d").date()
        except ValueError:
            return False
    return game_date <= published_date <= game_date + timedelta(days=1)


def matches_game(title: str, away: dict, home: dict) -> bool:
    """Comprobar equipos y marcador de ESPN, sin depender del orden ganador/local."""
    t = normalized(title)
    if not t.startswith("game recap "):
        return False
    for team in (away, home):
        name = normalized(team["name"])
        if not name:
            return False
        pattern = rf"\b{re.escape(name)}\b"
        if team.get("score") is not None:
            pattern += rf"\s+{re.escape(str(team['score']))}\b"
        if not re.search(pattern, t):
            return False
    return True


def get_games(game_date: date) -> list[dict]:
    response = requests.get(
        ESPN_SCOREBOARD,
        params={"dates": game_date.strftime("%Y%m%d"), "limit": 100},
        timeout=25,
    )
    response.raise_for_status()
    games = []
    for event in response.json().get("events", []):
        if not event.get("status", {}).get("type", {}).get("completed", False):
            continue  # Aún no terminó o fue suspendido
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        sides = {}
        for competitor in competitions[0].get("competitors", []):
            team = competitor.get("team", {})
            side = competitor.get("homeAway")
            if side in ("home", "away"):
                sides[side] = {
                    "name": team.get("name", ""),  # Lakers, Trail Blazers, 76ers...
                    "full": team.get("displayName", ""),
                    "abbr": team.get("abbreviation", ""),
                    "score": competitor.get("score"),
                }
        if "home" in sides and "away" in sides:
            games.append({"id": event.get("id", ""), **sides})
    return games


def find_recap_video(
    game: dict, game_date: date, search_results: int, diagnostics: bool = False
) -> dict | None:
    away, home = game["away"], game["home"]
    query = f'Game Recap {away["name"]} {away.get("score", "")} {home["name"]} {home.get("score", "")}'
    search_url = f"https://www.youtube.com/channel/{CHANNEL_ID}/search?" + urlencode(
        {"query": query}
    )
    search_options = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": search_results,
    }
    with YoutubeDL(search_options) as ydl:
        search = ydl.extract_info(search_url, download=False)

    if search is None:
        raise RuntimeError("YouTube no devolvió una respuesta de búsqueda")
    entries = [
        entry for entry in search.get("entries", []) if entry and entry.get("id")
    ]
    if diagnostics:
        print(f"    Búsqueda en Motion Station: {query}")
        print(f"    Resultados: {len(entries)}")

    candidates = []
    for entry in entries:
        if not matches_game(entry.get("title", ""), away, home):
            continue
        if entry.get("channel_id") not in (None, CHANNEL_ID):
            continue
        candidates.append(entry)

    # Preferir el resumen estándar frente a las variantes EXTENDED.
    candidates.sort(
        key=lambda x: (
            "extended" in normalized(x.get("title", "")),
            -int(x.get("view_count") or 0),
        )
    )
    if diagnostics:
        print(f"    Coinciden con equipos y marcador: {len(candidates)}")
        if not candidates:
            print(
                "    La ausencia en la búsqueda no demuestra que el vídeo no exista. "
                "YouTube puede ocultar vídeos restringidos por país."
            )
    # Obtener los metadatos aunque no haya formatos disponibles (p. ej., por país).
    # La descarga sigue comprobando la disponibilidad con sus propias opciones.
    check_options = {
        "quiet": True,
        "skip_download": True,
        "ignore_no_formats_error": True,
    }
    verification_errors = []
    with YoutubeDL(check_options) as ydl:
        for entry in candidates:
            url = f'https://www.youtube.com/watch?v={entry["id"]}'
            try:
                info = ydl.extract_info(url, download=False)
            except DownloadError as exc:
                verification_errors.append(str(exc))
                if diagnostics:
                    print(f"    No se pudo verificar {url}: {exc}")
                continue
            if not info or info.get("channel_id") != CHANNEL_ID:
                continue
            if not matches_video_date(info, game_date):
                if diagnostics:
                    print(
                        f"    Descartado por fecha de publicación: {info.get('upload_date')} | {url}"
                    )
                continue
            if matches_game(info.get("title", ""), away, home):
                return {"id": info["id"], "title": info["title"], "url": url}
    if verification_errors:
        raise RuntimeError(
            "No se pudieron verificar todos los candidatos de YouTube. "
            f"Último error: {verification_errors[-1]}"
        )
    return None


def download_video(url: str, destination: Path, stem: str, archive: Path) -> None:
    options = {
        "format": "bv*[height<=720]+ba/b[height<=720]/best",
        "outtmpl": str(destination / (stem + "_%(id)s.%(ext)s")),
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
        "download_archive": str(archive),
        "noplaylist": True,
        "retries": 3,
    }
    with YoutubeDL(options) as ydl:
        ydl.download([url])


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "Instala ffmpeg y ffprobe y añádelos al PATH para guardar vídeo/audio"
        )


def validate_audio_path(audio_path: Path) -> str:
    audio_format = audio_path.suffix.lstrip(".").lower()
    if audio_format not in {"mp3", "m4a", "wav"}:
        raise ValueError("El archivo de audio debe tener extensión .mp3, .m4a o .wav")
    return audio_format


def download_audio(url: str, audio_path: Path) -> Path:
    """Descargar únicamente una pista de audio; nunca usar un formato con vídeo."""
    audio_format = validate_audio_path(audio_path)
    if audio_path.is_file() and audio_path.stat().st_size:
        return audio_path
    require_ffmpeg()
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio",
        "outtmpl": str(audio_path.with_suffix(".%(ext)s")),
        "final_ext": audio_format,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ],
        # El archivo final controla las repeticiones. El historial de vídeos no
        # debe impedir descargar el audio de un vídeo descargado previamente.
        "noplaylist": True,
        "retries": 3,
    }
    with YoutubeDL(options) as ydl:
        ydl.download([url])
    if not audio_path.is_file() or not audio_path.stat().st_size:
        raise RuntimeError(f"No se generó el archivo de audio: {audio_path}")
    return audio_path


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extraer el audio de un vídeo local conservando el vídeo original."""
    audio_format = validate_audio_path(audio_path)
    if video_path.resolve() == audio_path.resolve():
        raise ValueError("El archivo de audio debe ser distinto del vídeo de origen")
    if not video_path.is_file():
        raise FileNotFoundError(f"No existe el vídeo: {video_path}")
    if audio_path.is_file() and audio_path.stat().st_size:
        return audio_path
    require_ffmpeg()
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    codecs = {"mp3": "libmp3lame", "m4a": "aac", "wav": "pcm_s16le"}
    with tempfile.NamedTemporaryFile(
        suffix=f".{audio_format}", dir=audio_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path.resolve()),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            codecs[audio_format],
        ]
        if audio_format != "wav":
            command.extend(["-b:a", "192k"])
        command.append(str(temporary_path.resolve()))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"No se pudo extraer el audio de {video_path}: {result.stderr.strip()}"
            )
        temporary_path.replace(audio_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return audio_path


def save_recap(
    video: dict,
    destination: Path,
    stem: str,
    archive: Path,
    mode: str,
    audio_format: str,
) -> dict:
    """Guardar los archivos elegidos y devolver sus rutas para el CSV."""
    if mode not in {"audio", "video", "both"}:
        raise ValueError("El modo de descarga debe ser audio, video o both")
    basename = f"{stem}_{video['id']}"
    video_path = destination / f"{basename}.mp4"
    audio_path = destination / f"{basename}.{audio_format}"
    paths = {"video_path": "", "audio_path": ""}
    if mode in {"video", "both"}:
        if not video_path.is_file() or not video_path.stat().st_size:
            download_video(video["url"], destination, stem, archive)
        if not video_path.is_file() or not video_path.stat().st_size:
            raise RuntimeError(
                f"No se generó el vídeo: {video_path}. Revisa el historial {archive}"
            )
        paths["video_path"] = str(video_path.resolve())
    if mode == "audio":
        paths["audio_path"] = str(download_audio(video["url"], audio_path).resolve())
    elif mode == "both":
        paths["audio_path"] = str(extract_audio(video_path, audio_path).resolve())
    return paths


# %% Preparar fecha y carpetas
if DOWNLOAD_MODE not in {"audio", "video", "both", "local_audio"}:
    raise ValueError("DOWNLOAD_MODE debe ser audio, video, both o local_audio")
if AUDIO_FORMAT not in {"mp3", "m4a", "wav"}:
    raise ValueError("AUDIO_FORMAT debe ser mp3, m4a o wav")
if DOWNLOAD_MODE == "local_audio" and not LOCAL_VIDEO:
    raise ValueError("Indica la ruta del vídeo en LOCAL_VIDEO")
try:
    game_date = date.fromisoformat(DATE)
except ValueError as exc:
    raise ValueError(
        "DATE debe tener formato AAAA-MM-DD, por ejemplo 2026-10-28"
    ) from exc
if SEARCH_RESULTS < 1:
    raise ValueError("SEARCH_RESULTS debe ser >= 1")

destination = Path(OUTPUT) / game_date.isoformat()
destination.mkdir(parents=True, exist_ok=True)
archive = Path(OUTPUT) / "downloaded.txt"

# %% Extraer de un vídeo local (modo local_audio; se guarda en destination)
if DOWNLOAD_MODE == "local_audio":
    local_video_path = Path(LOCAL_VIDEO).expanduser()
    local_audio_path = destination / f"{local_video_path.stem}.{AUDIO_FORMAT}"
    if LIST_ONLY:
        print(f"Audio previsto: {local_audio_path}")
    else:
        extract_audio(local_video_path, local_audio_path)
        print(f"Audio guardado en: {local_audio_path.resolve()}")

# %% Consultar partidos finalizados
games = []
if DOWNLOAD_MODE != "local_audio":
    if not LIST_ONLY:
        require_ffmpeg()
    try:
        games = get_games(game_date)
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"No se pudo consultar ESPN: {exc}") from exc
    print(
        f"Fecha NBA: {game_date} | Partidos finalizados: {len(games)} | Modo: {DOWNLOAD_MODE}"
    )

# %% Buscar resúmenes y descargarlos si LIST_ONLY es False
rows = []
for i, game in enumerate(games, 1):
    away, home = game["away"], game["home"]
    print(f'[{i}/{len(games)}] {away["full"]} @ {home["full"]}')
    row = {
        "date": game_date.isoformat(),
        "espn_game_id": game["id"],
        "away": away["full"],
        "home": home["full"],
        "title": "",
        "url": "",
        "status": "not_found",
        "video_path": "",
        "audio_path": "",
    }
    try:
        video = find_recap_video(game, game_date, SEARCH_RESULTS, DIAGNOSTICS)
        if video:
            row.update(
                {
                    "title": video["title"],
                    "url": video["url"],
                    "status": "found",
                }
            )
            print(f'    {video["url"]}')
            if not LIST_ONLY:
                stem = f'{game_date}_{away["abbr"]}_at_{home["abbr"]}'
                row.update(
                    save_recap(
                        video, destination, stem, archive, DOWNLOAD_MODE, AUDIO_FORMAT
                    )
                )
                row["status"] = "downloaded"
                for field in ("video_path", "audio_path"):
                    if row[field]:
                        print(f"    {field}: {row[field]}")
        else:
            print(
                "    No se encontró un resumen de Motion Station con equipos, marcador y fecha coincidentes"
            )
    except Exception as exc:
        row["status"] = "error"
        print(f"    ERROR: {exc}")
    rows.append(row)

# %% Guardar el CSV
if DOWNLOAD_MODE != "local_audio":
    csv_path = destination / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV guardado en: {csv_path}")
    print(f'Resúmenes localizados: {sum(bool(r["url"]) for r in rows)}/{len(games)}')
