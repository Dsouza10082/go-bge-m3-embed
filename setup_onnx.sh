#!/bin/bash

echo "Downloading ONNX files from Google Drive..."

mkdir -p onnx

if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    python3 -m pip install gdown
fi

declare -a FILE_IDS=(
    "1mZYHCIOQpXijmR5L6Tk6nfo7jv5DDaPk"
    "1xoYEG7Fa5UJYOfTobv2TTE_O4OF1yB1H"
    "1k9UsvfmsibOB2uzv2JYchn9Ngoa8Eq2b"
    "1JoIH1efT776eeTRZyNUBJ8Rl9ROs2L8b"
    "1-GG5QGACbj0KRec_sElR8meyEx239Gc8"
    "1Wqv-1gTJIVLM8qvB2IQXgHjn0Zf_kHUF"
    "1PJ5s31ZgcMYeZxCEhXIBW1xD6uQ_q3Mx"
    "1SRfvlU-vjFWkgONjCtw-czRw8cfcJrL7"
    "1iQ6N667s8CzyNn6gHTFvDgqo21omxxyR"
    "1LlPQqCICxudy13f-71ridgZcoxxr7GZ0"
    "1q2UapZoi3xbMMK3-kgBkYoypipHdFMGt"
    "1GXDiCpXp3XPvJ70oTiJyEvP6R1B4Iny3"
    "1Jwyln7dnzNeh3JUEqX7ioIWRkyCZlM49"
    "1oHXX-Knr1yD8uUtkKc2yDSXgk7df3EzL"
    "1saWmizioOOg8riG-BY-QCp13agpW-zRm"
    "10bwiG2PL6Z49bIiks3Ey--QscbcDtvdV"
    "1L1xggWsb_-BHuX7H3DG5FNn2YQA0udT_"
    "1JoIH1efT776eeTRZyNUBJ8Rl9ROs2L8b"
)

declare -a FILE_NAMES=(
    "libonnxruntime.dylib"
    "model.onnx"
    "model.onnx_data"
    "sentencepiece.bpe.model"
    "tokenizer_config.json"
    "tokenizer.json"
    "libonnxruntime_providers_shared.so"
    "libonnxruntime.pc"
    "libonnxruntime.so"
    "libonnxruntime.so.1"
    "libonnxruntime.so.1.23.1"
    "onnxruntime_providers_shared.dll"
    "onnxruntime_providers_shared.lib"
    "onnxruntime_providers_shared.pdb"
    "onnxruntime.dll"
    "onnxruntime.lib"
    "onnxruntime.pdb"
    "sentencepiece.bpe.model"
)

for i in "${!FILE_IDS[@]}"; do
    FILE_ID="${FILE_IDS[$i]}"
    FILE_NAME="${FILE_NAMES[$i]}"
    OUTPUT_PATH="onnx/${FILE_NAME}"
    
    echo "Downloading ${FILE_NAME}..."

    gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT_PATH}"

    if [ -f "${OUTPUT_PATH}" ]; then
        echo "✓ ${FILE_NAME} downloaded successfully"
    else
        echo "✗ Error downloading ${FILE_NAME}"
        echo "  You can download manually from: https://drive.google.com/file/d/${FILE_ID}/view"
        echo "  Save as: ${OUTPUT_PATH}"
    fi
done

echo ""
echo "Download completed!"
