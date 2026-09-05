import hashlib
import json
import os
from datetime import datetime, timezone


# ---------------------------------------------------------
# Blockchain storage file
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

BLOCKCHAIN_FILE = os.path.join(
    BASE_DIR,
    "blockchain_data.json"
)


# ---------------------------------------------------------
# Hash function
# ---------------------------------------------------------

def calculate_hash(data):
    """
    Creates SHA-256 hash for the supplied data.
    """

    encoded_data = json.dumps(
        data,
        sort_keys=True
    ).encode("utf-8")

    return hashlib.sha256(encoded_data).hexdigest()


# ---------------------------------------------------------
# Create Genesis Block
# ---------------------------------------------------------

def create_genesis_block():
    """
    Creates the first block of the blockchain.
    """

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    block = {
        "index": 0,
        "timestamp": timestamp,
        "type": "GENESIS",
        "evidence": {
            "message": "Voice Impersonation Protection Blockchain"
        },
        "previous_hash": "0",
    }

    block["hash"] = calculate_hash(block)

    return block


# ---------------------------------------------------------
# Load Blockchain
# ---------------------------------------------------------

def load_blockchain():
    """
    Loads blockchain from local JSON file.
    """

    if not os.path.exists(BLOCKCHAIN_FILE):
        blockchain = [
            create_genesis_block()
        ]

        save_blockchain(blockchain)

        return blockchain

    try:
        with open(
            BLOCKCHAIN_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):

        blockchain = [
            create_genesis_block()
        ]

        save_blockchain(blockchain)

        return blockchain


# ---------------------------------------------------------
# Save Blockchain
# ---------------------------------------------------------

def save_blockchain(blockchain):
    """
    Saves blockchain to JSON file.
    """

    with open(
        BLOCKCHAIN_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            blockchain,
            file,
            indent=4
        )


# ---------------------------------------------------------
# Add Evidence Block
# ---------------------------------------------------------

def add_evidence_block(
    filename,
    voice_label,
    voice_confidence,
    fraud_score,
    fraud_level,
    overall_score,
    overall_risk_level,
    transcript,
):
    """
    Adds an audio analysis result as a new blockchain block.
    """

    blockchain = load_blockchain()

    previous_block = blockchain[-1]

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    evidence = {
        "filename": filename,
        "voice_label": voice_label,
        "voice_confidence": round(
            voice_confidence,
            4
        ),
        "fraud_score": fraud_score,
        "fraud_level": fraud_level,
        "overall_score": overall_score,
        "overall_risk_level": overall_risk_level,
        "transcript": transcript,
    }

    block = {
        "index": len(blockchain),
        "timestamp": timestamp,
        "type": "VOICE_ANALYSIS",
        "evidence": evidence,
        "previous_hash": previous_block["hash"],
    }

    block["hash"] = calculate_hash(block)

    blockchain.append(block)

    save_blockchain(blockchain)

    return block


# ---------------------------------------------------------
# Verify Blockchain
# ---------------------------------------------------------

def verify_blockchain():
    """
    Verifies whether the blockchain has been tampered with.
    """

    blockchain = load_blockchain()

    for index, block in enumerate(blockchain):

        # Verify current block hash
        stored_hash = block["hash"]

        block_without_hash = {
            key: value
            for key, value in block.items()
            if key != "hash"
        }

        calculated_hash = calculate_hash(
            block_without_hash
        )

        if stored_hash != calculated_hash:
            return {
                "valid": False,
                "message": f"Block {index} hash mismatch."
            }

        # Verify previous block connection
        if index > 0:

            previous_block = blockchain[index - 1]

            if block["previous_hash"] != previous_block["hash"]:
                return {
                    "valid": False,
                    "message": (
                        f"Block {index} previous hash mismatch."
                    )
                }

    return {
        "valid": True,
        "message": "Blockchain integrity verified."
    }


# ---------------------------------------------------------
# Get Blockchain
# ---------------------------------------------------------

def get_blockchain():
    """
    Returns the complete blockchain.
    """

    return load_blockchain()