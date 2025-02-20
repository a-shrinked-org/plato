#!/bin/bash

set -e

# Parse arguments
URL="$1"
LANG="en"
VERBOSE="false"
IMAGES="false"
MODEL="gemini"

while [[ $# -gt 0 ]]; do
    case $1 in
    --lang) LANG="$2"; shift 2 ;;
    --verbose) VERBOSE="true"; shift ;;
    --images) IMAGES="true"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    *) shift ;;
    esac
done

# Basic validation
if [ -z "$URL" ]; then
    echo "Error: URL is required"
    exit 1
fi

# Configure model
case "$MODEL" in
"gemini")
    if [ -z "$GOOGLE_CLOUD_PROJECT" ] || [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "Error: Missing Gemini credentials"
        exit 1
    fi
    MODEL_FLAG="--model gemini"
    ;;
"anthropic")
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "Error: ANTHROPIC_API_KEY not set"
        exit 1
    fi
    MODEL_FLAG="--anthropic-api-key $ANTHROPIC_API_KEY"
    ;;
*)
    echo "Error: Unsupported model: $MODEL"
    exit 1
    ;;
esac

echo "Starting conversion..."
echo "URL: $URL"

# Single transcription step - no status check needed
if [ -z "$ASSEMBLYAI_API_KEY" ]; then
    echo "Getting text from URL..."
    plato "$URL" ${IMAGES:+--images} --lang "$LANG" $MODEL_FLAG
else
    echo "Transcribing with AssemblyAI..."
    plato "$URL" ${IMAGES:+--images} --assemblyai-api-key "$ASSEMBLYAI_API_KEY" --lang "$LANG" $MODEL_FLAG
fi

# One-shot content generation
echo "Generating content..."
TITLE=$(plato --title "$URL" --lang "$LANG" $MODEL_FLAG)
ABSTRACT=$(plato --abstract "$URL" --lang "$LANG" $MODEL_FLAG)
PASSAGES=$(plato --passages --chapters --inline-references "$URL" --lang "$LANG" $MODEL_FLAG)
CHAPTERS=$(plato --chapters "$URL" --lang "$LANG" $MODEL_FLAG)
REFERENCES=$(plato --references "$URL" --lang "$LANG" $MODEL_FLAG)

# Generate output files
echo "Creating documents..."
generate_document() {
    local include_refs=$1
    echo $'# '"${TITLE}"$'\n'
    echo $'## Origin\n\n'"$URL"$'\n'
    echo $'## Abstract\n\n'"${ABSTRACT}"$'\n'
    echo $'## Chapters\n\n'"$CHAPTERS"$'\n'
    echo $'## Discussion\n\n'"$PASSAGES"$'\n'
    [ "$include_refs" = true ] && echo $'## References\n\n'"$REFERENCES"$'\n'
}

sanitized_title=$(echo "$TITLE" | sed 's/[^a-zA-Z0-9]/_/g')

echo "Generating PDF files..."
generate_document false | pandoc -o "${sanitized_title}-no-refs.pdf" --from markdown --pdf-engine=xelatex
generate_document true | pandoc -o "${sanitized_title}-refs.pdf" --from markdown+header_attributes --pdf-engine=xelatex

if [ "$VERBOSE" = true ]; then
    echo "<title>"
    echo "$TITLE"
    echo "</title>"
    echo
    echo "<abstract>"
    echo "$ABSTRACT"
    echo "</abstract>"
fi

echo "Done!"