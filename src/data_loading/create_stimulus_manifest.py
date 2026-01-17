"""
Create stimulus_manifest.csv mapping audio filenames to transcripts and duration.
"""

import logging
import subprocess
import warnings
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import whisper
from mutagen import File
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress Whisper FP16 warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Configurable audio extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

# Manifest column names
MANIFEST_COLUMNS = [
    "stimulus_id",
    "filename",
    "filepath",
    "transcript_text",
    "duration_seconds",
    "sample_rate",
    "channels",
    "stimulus_type",
    "word_count",
    "notes",
]


def _has_valid_metadata(metadata: Dict) -> bool:
    """Check if metadata contains all required fields."""
    return all(
        metadata.get(key) is not None and metadata.get(key) > 0
        for key in ["duration_seconds", "sample_rate", "channels"]
    )


def _remux_audio_file(filepath: Path) -> bool:
    """
    Re-mux audio file with ffmpeg to fix metadata structure.
    Replaces original file in-place without creating backup files.
    """
    try:
        fixed_path = filepath.with_stem(f"{filepath.stem}_fixed")
        
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", filepath,
                "-c", "copy",
                "-movflags", "+faststart",
                fixed_path,
                "-y",
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.warning(f"ffmpeg failed for {filepath.name}: {result.stderr}")
            return False
        
        # Replace original with fixed file
        filepath.unlink()
        fixed_path.rename(filepath)
        
        logger.info(f"Re-muxed {filepath.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to re-mux {filepath.name}: {e}")
        return False


def _extract_stimulus_metadata(filepath: Path) -> Dict:
    """
    Extract metadata from audio file using mutagen.
    If metadata extraction fails, re-mux the file with ffmpeg and try to extract again.
    """
    def _try_extract(file_path: Path) -> Dict:
        """Attempt to extract metadata from file."""
        try:
            audio = File(file_path)
            if audio is None or not hasattr(audio, "info") or audio.info is None:
                return {
                    "duration_seconds": None,
                    "sample_rate": None,
                    "channels": None,
                }
            
            return {
                "duration_seconds": getattr(audio.info, "length", None),
                "sample_rate": getattr(audio.info, "sample_rate", None),
                "channels": getattr(audio.info, "channels", None),
            }
        except Exception as e:
            logger.debug(f"Metadata extraction failed for {file_path.name}: {e}")
            return {
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
            }
    
    metadata = _try_extract(filepath)
    if not _has_valid_metadata(metadata):
        logger.info(f"Incomplete metadata for {filepath.name}, attempting re-mux...")
        if _remux_audio_file(filepath):
            metadata = _try_extract(filepath)

    return metadata


def _get_stimulus_type_from_folder(folder_name: str) -> str:
    """Map folder name to stimulus type."""
    folder_lower = folder_name.lower()
    if "sentence" in folder_lower:
        return "language_sentence"
    elif "prompt" in folder_lower:
        return "prompt"
    elif "static" in folder_lower:
        return "command"
    elif "voice" in folder_lower:
        return "control_voice"
    else:
        return "unknown"


def _extract_stimulus_id(filepath: Path) -> str:
    """Extract stimulus ID from filename (e.g., lang0, lang1)."""
    return filepath.stem


def _transcribe_audio(filepath: Path, model: Any):
    """Transcribe audio using Whisper and extract transcript.
    
    Note: Word timestamps are disabled for faster processing. To enable word-level
    timestamps, set word_timestamps=True and uncomment segments/words extraction.
    Word timestamps are complex to store in CSV - consider saving to a separate JSON file.
    """
    try:
        result = model.transcribe(
            str(filepath),
            word_timestamps=False,  # Set to True if word-level timestamps are needed
            fp16=False,  # Explicit FP32 for CPU
        )
        transcript = result.get("text", "").strip()
        
        # Word timestamps extraction
        # segments = result.get("segments", [])
        # words = [segment["words"] for segment in segments if "words" in segment]

        return {
            "transcript_text": transcript,
            "word_count": len(transcript.split()) if transcript else 0,
            # "segments": segments,
            # "words": words,
        }
    except Exception as e:
        logger.warning(f"Error transcribing {filepath.name}: {e}")
        return {
            "transcript_text": "",
            "word_count": 0,
            # "segments": [],
            # "words": [],
        }


def _process_single_file(filepath: Path, stimulus_type_map: Dict, audio_folder: Path, model: Any):
    """Process a single file (metadata + transcription)."""
    metadata = _extract_stimulus_metadata(filepath)
    transcript_info = _transcribe_audio(filepath, model)

    return {
        "stimulus_id": _extract_stimulus_id(filepath),
        "filename": filepath.name,
        "filepath": (
            filepath.relative_to(audio_folder.parent)
            if audio_folder.parent in filepath.parents
            else filepath
        ),
        "transcript_text": transcript_info["transcript_text"],
        "duration_seconds": metadata["duration_seconds"],
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
        "stimulus_type": stimulus_type_map[filepath],
        "word_count": transcript_info["word_count"],
        "notes": "",
    }


def _scan_audio_files(audio_folder: Path):
    """Scan audio folder and return files with stimulus type mapping."""
    all_files = []
    stimulus_type_map = {}

    for level1_folder in audio_folder.iterdir():
        if not level1_folder.is_dir():
            continue

        folder_files = list(
            chain.from_iterable(
                level1_folder.rglob(f"*{ext}") for ext in AUDIO_EXTENSIONS
            )
        )
        all_files.extend(folder_files)

        stimulus_type = _get_stimulus_type_from_folder(level1_folder.name)
        stimulus_type_map.update({f: stimulus_type for f in folder_files})

    return sorted(all_files), stimulus_type_map


def _load_whisper_model(whisper_model_name: str):
    """Load and return Whisper model."""
    logger.info(f"Loading `{whisper_model_name}` size Whisper model for transcription")
    model = whisper.load_model(whisper_model_name)
    logger.info("Whisper model for transcription loaded successfully")
    return model


def create_stimulus_manifest(
    audio_folder: Path,
    output_path: Path,
    whisper_model_name: str = "base",
) -> pd.DataFrame:
    """
    Create stimulus manifest CSV from all audio files in the folder.

    Parameters
    ----------
    audio_folder : Path
        Root folder containing audio files (will scan recursively)
    output_path : Path
        Path to save the stimulus_manifest.csv
    whisper_model_name : str
        Whisper model size (tiny, base, small, medium, large)
    """
    logger.info(f"Scanning audio files in: {audio_folder}")
    all_files, stimulus_type_map = _scan_audio_files(audio_folder)
    logger.info(f"Found {len(all_files)} audio files")

    if not all_files:
        logger.warning("No audio files found. Creating empty manifest.")
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    model = _load_whisper_model(whisper_model_name)

    logger.info(f"Processing {len(all_files)} audio files (transcription + metadata)")
    manifest_data = []
    for filepath in tqdm(all_files, desc="Processing audio files", unit="file"):
        try:
            result = _process_single_file(filepath, stimulus_type_map, audio_folder, model)
            manifest_data.append(result)
        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")

    manifest_df = pd.DataFrame(manifest_data, columns=MANIFEST_COLUMNS)
    # Sort by stimulus_id for language files, maintain order for others
    manifest_df = manifest_df.sort_values("stimulus_id", kind="mergesort")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False)
    logger.info(f"Saved manifest with {len(manifest_df)} entries to: {output_path}")

    return manifest_df


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    audio_folder = project_root / "data" / "Audio"
    output_path = project_root / "data" / "processed" / "stimulus_manifest.csv"

    create_stimulus_manifest(audio_folder, output_path)


if __name__ == "__main__":
    main()
