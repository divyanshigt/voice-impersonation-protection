import os
import tempfile
import subprocess

import torch
import whisper
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ai_layer.audio_features import load_and_preprocess
from app.ai_layer.mel_features import extract_mel_features
from app.ai_layer.model_architecture import CloneDetectorCNN
from app.cyber_layer.fraud_detector import analyze_fraud
from app.cyber_layer.risk_engine import calculate_overall_risk
from app.blockchain_layer.evidence_logger import (
    add_evidence_block,
    verify_blockchain,
    get_blockchain,
)


router = APIRouter(
    prefix="/api/audio",
    tags=["Audio"],
)


# ============================================================
# SECURITY CONFIGURATION
# ============================================================

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
}


# ============================================================
# BLOCKCHAIN ENDPOINTS
# ============================================================

@router.get("/blockchain/verify")
def verify_blockchain_endpoint():
    return verify_blockchain()


@router.get("/blockchain")
def get_blockchain_endpoint():
    return {
        "blockchain": get_blockchain()
    }


# ============================================================
# AI VOICE DETECTION MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    ),
    "models",
    "clone_detector.pt",
)


device = torch.device("cpu")

model = CloneDetectorCNN(
    dropout=0.3
).to(device)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)


if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)


model.eval()


# ============================================================
# WHISPER MODEL
# ============================================================

print("Loading Whisper model...")

whisper_model = whisper.load_model("small")

print("Whisper model loaded successfully.")


# ============================================================
# MODEL STATUS
# ============================================================

@router.get("/model-status")
def model_status():

    return {
        "model_loaded": True,
        "model_name": "CloneDetectorCNN",
        "device": str(device),
        "speech_to_text": True,
        "whisper_model": "small",
    }


# ============================================================
# AUDIO PROCESS
# ============================================================

@router.post("/process")
async def process_audio(
    file: UploadFile = File(...)
):

    # ========================================================
    # BASIC FILE VALIDATION
    # ========================================================

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )


    # Only use the filename itself.
    # Prevents paths such as ../../something.
    filename = os.path.basename(
        file.filename
    )


    extension = os.path.splitext(
        filename
    )[1].lower()


    # ========================================================
    # EXTENSION VALIDATION
    # ========================================================

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Allowed formats: WAV, MP3, M4A, FLAC, OGG, AAC."
            ),
        )


    original_extension = extension


    temp_path = None
    converted_path = None


    try:

        # ====================================================
        # SAVE UPLOADED FILE SAFELY
        # ====================================================

        total_size = 0

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temp_path = temp_file.name


            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break


                total_size += len(chunk)


                # --------------------------------------------
                # FILE SIZE LIMIT
                # --------------------------------------------

                if total_size > MAX_FILE_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File size exceeds the "
                            "25 MB limit."
                        ),
                    )


                temp_file.write(chunk)


        # ====================================================
        # EMPTY FILE CHECK
        # ====================================================

        if total_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty.",
            )


        # ====================================================
        # AAC → WAV CONVERSION
        # ====================================================

        processing_path = temp_path


        if extension == ".aac":

            print(
                "AAC file detected. "
                "Converting to WAV..."
            )


            converted_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
            )


            converted_path = (
                converted_file.name
            )


            converted_file.close()


            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    converted_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )


            processing_path = converted_path


            print(
                "AAC → WAV conversion successful."
            )


        # ====================================================
        # AUDIO PREPROCESSING
        # ====================================================

        processed_audio = load_and_preprocess(
            processing_path
        )


        # ====================================================
        # AI VOICE DETECTION
        # ====================================================

        mel_features = extract_mel_features(
            processing_path
        )


        input_tensor = torch.tensor(
            mel_features,
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0).to(device)


        with torch.no_grad():

            logits = model(
                input_tensor
            )

            spoof_probability = (
                torch.sigmoid(logits).item()
            )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if spoof_probability >= 0.5:

            label = "spoof"

            confidence = spoof_probability

        else:

            label = "bonafide"

            confidence = (
                1 - spoof_probability
            )


        # ====================================================
        # SPEECH TO TEXT
        # ====================================================

        print(
            f"Transcribing: {filename}"
        )


        transcription_result = (
            whisper_model.transcribe(
                processing_path,
                task="transcribe",
                fp16=False,
            )
        )


        transcript = (
            transcription_result
            .get("text", "")
            .strip()
        )


        language_code = (
            transcription_result.get(
                "language",
                "unknown"
            )
        )


        # ====================================================
        # LANGUAGE MAPPING
        # ====================================================

        language_names = {

            "hi": "Hindi",

            "en": "English",

            "fr": "French",

            "es": "Spanish",

            "de": "German",

            "ja": "Japanese",

            "zh": "Chinese",

            "ur": "Urdu",

            "pa": "Punjabi",

            "bn": "Bengali",

            "mr": "Marathi",

            "gu": "Gujarati",

            "ta": "Tamil",

            "te": "Telugu",

            "kn": "Kannada",

            "ml": "Malayalam",

        }


        detected_language = (
            language_names.get(
                language_code,
                language_code
            )
        )


        # ====================================================
        # FRAUD DETECTION
        # ====================================================

        fraud_analysis = analyze_fraud(
            voice_label=label,
            voice_confidence=confidence,
            transcript=transcript,
        )


        # ====================================================
        # OVERALL RISK ENGINE
        # ====================================================

        risk_analysis = calculate_overall_risk(
            voice_label=label,
            voice_confidence=confidence,
            fraud_score=fraud_analysis[
                "risk_score"
            ],
        )


        # ====================================================
        # BLOCKCHAIN EVIDENCE
        # ====================================================

        blockchain_block = add_evidence_block(

            filename=filename,

            voice_label=label,

            voice_confidence=confidence,

            fraud_score=fraud_analysis[
                "risk_score"
            ],

            fraud_level=fraud_analysis[
                "risk_level"
            ],

            overall_score=risk_analysis[
                "overall_score"
            ],

            overall_risk_level=risk_analysis[
                "risk_level"
            ],

            transcript=transcript,
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "filename": filename,


            "audio": {

                "original_format":
                    original_extension.replace(
                        ".",
                        ""
                    ),

                "sample_rate":
                    processed_audio.sample_rate,

                "channels":
                    processed_audio.channels,

                "duration":
                    processed_audio.duration,
            },


            "prediction": {

                "label":
                    label,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "spoof_probability":
                    round(
                        spoof_probability,
                        4
                    ),
            },


            "speech_to_text": {

                "transcript":
                    transcript,

                "language":
                    detected_language,
            },


            "fraud_detection":
                fraud_analysis,


            "risk_analysis":
                risk_analysis,


            "blockchain": {

                "block_index":
                    blockchain_block[
                        "index"
                    ],

                "block_hash":
                    blockchain_block[
                        "hash"
                    ],

                "previous_hash":
                    blockchain_block[
                        "previous_hash"
                    ],

                "timestamp":
                    blockchain_block[
                        "timestamp"
                    ],
            },


            "model": {

                "name":
                    "CloneDetectorCNN",

                "version":
                    "Phase-2C",

            },


            "processed":
                True,
        }


    # ========================================================
    # HTTP EXCEPTIONS
    # ========================================================

    except HTTPException:

        # Preserve the original status code
        # instead of converting it into 500.
        raise


    # ========================================================
    # FFMPEG ERRORS
    # ========================================================

    except subprocess.TimeoutExpired:

        raise HTTPException(
            status_code=400,
            detail=(
                "Audio conversion timed out. "
                "The uploaded audio may be invalid or too complex."
            ),
        )


    except subprocess.CalledProcessError:

        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to process the audio file. "
                "Please upload a valid audio recording."
            ),
        )


    # ========================================================
    # GENERAL ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "Audio analysis failed. "
                "Please try another audio file."
            ),
        )


    # ========================================================
    # TEMPORARY FILE CLEANUP
    # ========================================================

    finally:

        if temp_path:

            try:

                if os.path.exists(
                    temp_path
                ):

                    os.remove(
                        temp_path
                    )

            except OSError as cleanup_error:

                print(
                    "Temporary file cleanup error:",
                    cleanup_error
                )


        if converted_path:

            try:

                if os.path.exists(
                    converted_path
                ):

                    os.remove(
                        converted_path
                    )

            except OSError as cleanup_error:

                print(
                    "Converted file cleanup error:",
                    cleanup_error
                )


        try:

            await file.close()

        except Exception:

            pass