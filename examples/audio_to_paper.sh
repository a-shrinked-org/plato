#!/bin/bash

set -e

# Constants
MAX_RETRIES=2
RETRY_DELAY=2

# Function to handle errors
handle_error() {
    local exit_code=$?
    echo "Error occurred in script at line $1, exit code: $exit_code"
    exit $exit_code
}

trap 'handle_error ${LINENO}' ERR

# Function for retrying commands with exponential backoff and content validation
get_with_retry() {
    local cmd="$1"
    local error_msg="$2"
    local content_type="$3"  # "title", "abstract", etc.
    local retries=0
    local output=""
    local max_wait=30  # Maximum wait time in seconds
    
    while ((retries < MAX_RETRIES)); do
        # First check if transcription/indexing is complete
        if ! plato --status "$URL" | grep -q "complete"; then
            echo "Content still processing, waiting..." >&2
            sleep $((min(RETRY_DELAY ** retries, max_wait)))
            ((retries++))
            continue
        fi
        
        output=$(eval "$cmd" 2>/dev/null || true)
        
        # Content-specific validation
        case "$content_type" in
            "title")
                if [ -n "$output" ] && [ "$output" != "Missing Title" ]; then
                    echo "$output"
                    return 0
                fi
                ;;
            "abstract")
                if [ -n "$output" ] && [ "$output" != "Missing Summary" ]; then
                    echo "$output"
                    return 0
                fi
                ;;
            *)
                if [ -n "$output" ]; then
                    echo "$output"
                    return 0
                fi
                ;;
        esac
        
        ((retries++))
        echo "Attempt $retries failed for $content_type, retrying in $((RETRY_DELAY ** retries)) seconds..." >&2
        sleep $((min(RETRY_DELAY ** retries, max_wait)))
    done
    
    # If we failed to get content, try to generate it
    case "$content_type" in
        "title")
            output=$(plato --generate-title "$URL" --lang "$LANG" $MODEL_FLAG 2>/dev/null || echo "$error_msg")
            ;;
        "abstract")
            output=$(plato --generate-summary "$URL" --lang "$LANG" $MODEL_FLAG 2>/dev/null || echo "$error_msg")
            ;;
    esac
    
    echo "${output:-$error_msg}"
    return 1
}

# Function to sanitize output
sanitize_output() {
    local input="$1"
    local default="$2"
    echo "${input:-$default}" | sed 's/[[:cntrl:]]//g' | tr -s ' '
}

# Parse command line arguments
URL="$1"
LANG="en"
VERBOSE="false"
IMAGES="false"
MODEL="anthropic"  # Changed default from "gemini" to "anthropic"

while [[ $# -gt 0 ]]; do
    case $1 in
    --lang)
        LANG="$2"
        shift 2
        ;;
    --verbose)
        VERBOSE="true"
        shift
        ;;
    --images)
        IMAGES="true"
        shift
        ;;
    --model)
        MODEL="$2"
        shift 2
        ;;
    *)
        shift
        ;;
    esac
done

# Validate inputs
if [ -z "$URL" ]; then
    echo "Error: URL is required"
    exit 1
fi

# Validate model selection and API keys
case "$MODEL" in
"anthropic")
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "Error: ANTHROPIC_API_KEY is not set"
        echo "Obtain it from https://console.anthropic.com/keys"
        echo "Run: export ANTHROPIC_API_KEY=<your-api-key>"
        exit 1
    fi
    MODEL_FLAG="--anthropic-api-key $ANTHROPIC_API_KEY"
    MODEL_INFO="Claude 3.5 Sonnet (Anthropic, 2024-06-20)"  # More specific version info
    ;;
"gemini")
    if [ -z "$GOOGLE_API_KEY" ]; then
        echo "Error: GOOGLE_API_KEY is not set"
        echo "Obtain it from https://makersuite.google.com/app/apikey"
        echo "Run: export GOOGLE_API_KEY=<your-api-key>"
        exit 1
    fi
    MODEL_FLAG="--gemini-api-key $GOOGLE_API_KEY"
    MODEL_INFO="Gemini Pro 2.0 (Google, 2024-02-05)"  # Updated model info
    ;;
*)
    echo "Error: Invalid model: $MODEL"
    echo "Available models: anthropic, gemini"
    exit 1
    ;;
esac

# Load language-specific prompts
case "$LANG" in
"en")
    CONTRIBUTORS_PROMPT="Thoroughly review the <context> and identify the list of contributors. Output as Markdown list: First Name, Last Name, Title, Organization. Output \"Unknown\" if the contributors are not known. In the end of the list always add \"- [Platogram](https://github.com/code-anyway/platogram), Chief of Stuff, Code Anyway, Inc.\". Start with \"## Contributors, Acknowledgements, Mentions\""
    CONTRIBUTORS_PREFILL=$'## Contributors, Acknowledgements, Mentions\n'
    INTRODUCTION_PROMPT="Thoroughly review the <context> and write \"Introduction\" chapter for the paper. Write in the style of the original <context>. Use only words from <context>. Use quotes from <context> when necessary. Make sure to include <markers>. Output as Markdown. Start with \"## Introduction\""
    INTRODUCTION_PREFILL=$'## Introduction\n'
    CONCLUSION_PROMPT="Thoroughly review the <context> and write \"Conclusion\" chapter for the paper. Write in the style of the original <context>. Use only words from <context>. Use quotes from <context> when necessary. Make sure to include <markers>. Output as Markdown. Start with \"## Conclusion\""
    CONCLUSION_PREFILL=$'## Conclusion\n'
    ;;
"es")
    CONTRIBUTORS_PROMPT="Revise a fondo el <context> e identifique la lista de contribuyentes. Salida como lista Markdown: Nombre, Apellido, Título, Organización. Salida \"Desconocido\" si los contribuyentes no se conocen. Al final de la lista, agregue siempre \"- [Platogram](https://github.com/code-anyway/platogram), Chief of Stuff, Code Anyway, Inc.\". Comience con \"## Contribuyentes, Agradecimientos, Menciones\""
    CONTRIBUTORS_PREFILL=$'## Contribuyentes, Agradecimientos, Menciones\n'
    INTRODUCTION_PROMPT="Revise a fondo el <context> y escriba el capítulo \"Introducción\" para el artículo. Escriba en el estilo del original <context>. Use solo las palabras de <context>. Use comillas del original <context> cuando sea necesario. Asegúrese de incluir <markers>. Salida como Markdown. Comience con \"## Introducción\""
    INTRODUCTION_PREFILL=$'## Introducción\n'
    CONCLUSION_PROMPT="Revise a fondo el <context> y escriba el capítulo \"Conclusión\" para el artículo. Escriba en el estilo del original <context>. Use solo las palabras de <context>. Use comillas del original <context> cuando sea necesario. Asegúrese de incluir <markers>. Salida como Markdown. Comience con \"## Conclusión\""
    CONCLUSION_PREFILL=$'## Conclusión\n'
    ;;
*)
    echo "Error: Unsupported language: $LANG"
    exit 1
    ;;
esac

echo "Indexing $URL..."
echo "IMAGES: $IMAGES"

# Handle audio transcription
if [ -z "$ASSEMBLYAI_API_KEY" ]; then
    echo "ASSEMBLYAI_API_KEY is not set. Retrieving text from URL (subtitles, etc)."
    if ! plato "$URL" ${IMAGES:+--images} --lang "$LANG" >/dev/null; then
        echo "Error: Failed to retrieve text from URL"
        exit 1
    fi
else
    echo "Transcribing audio to text using AssemblyAI..."
    if ! plato "$URL" ${IMAGES:+--images} --assemblyai-api-key "$ASSEMBLYAI_API_KEY" --lang "$LANG" >/dev/null; then
        echo "Error: Failed to transcribe audio"
        exit 1
    fi
fi

echo "Fetching title, abstract, passages, and references..."

# Check if file is still processing
echo "Checking content processing status..."
if ! plato --status "$URL" | grep -q "complete"; then
    echo "Waiting for content processing to complete..."
    sleep 5
fi

# Get content with retries and sanitization
echo "Retrieving title..."
TITLE=$(get_with_retry "plato --title '$URL' --lang '$LANG' $MODEL_FLAG" "Generated Title" "title")
echo "Retrieving abstract..."
ABSTRACT=$(get_with_retry "plato --abstract '$URL' --lang '$LANG' $MODEL_FLAG" "Generated Summary" "abstract")
echo "Retrieving passages..."
PASSAGES=$(get_with_retry "plato --passages --chapters --inline-references '$URL' --lang '$LANG' $MODEL_FLAG" "No content available" "passages")
echo "Retrieving references..."
REFERENCES=$(get_with_retry "plato --references '$URL' --lang '$LANG' $MODEL_FLAG" "No references available" "references")
echo "Retrieving chapters..."
CHAPTERS=$(get_with_retry "plato --chapters '$URL' --lang '$LANG' $MODEL_FLAG" "No chapters available" "chapters")

# Sanitize outputs
TITLE=$(sanitize_output "$TITLE" "Missing Title")
ABSTRACT=$(sanitize_output "$ABSTRACT" "Missing Summary")
PASSAGES=$(sanitize_output "$PASSAGES" "No content available")
REFERENCES=$(sanitize_output "$REFERENCES" "No references available")
CHAPTERS=$(sanitize_output "$CHAPTERS" "No chapters available")

echo "Generating Contributors..."
CONTRIBUTORS=$(plato \
    --query "$CONTRIBUTORS_PROMPT" \
    --generate \
    --context-size large \
    --prefill "$CONTRIBUTORS_PREFILL" \
    "$URL" --lang "$LANG" $MODEL_FLAG || echo "Unknown Contributors")

echo "Generating Introduction..."
INTRODUCTION=$(plato \
    --query "$INTRODUCTION_PROMPT" \
    --generate \
    --context-size large \
    --inline-references \
    --prefill "$INTRODUCTION_PREFILL" \
    "$URL" --lang "$LANG" $MODEL_FLAG || echo "No introduction available")

echo "Generating Conclusion..."
CONCLUSION=$(plato \
    --query "$CONCLUSION_PROMPT" \
    --generate \
    --context-size large \
    --inline-references \
    --prefill "$CONCLUSION_PREFILL" \
    "$URL" --lang "$LANG" $MODEL_FLAG || echo "No conclusion available")

# Function to generate document
generate_document() {
    local include_refs=$1
    local output_suffix=$2
    
    (
        echo $'# '"${TITLE}"$'\n'
        echo $'## Origin\n\n'"$URL"$'\n'
        echo $'## Abstract\n\n'"${ABSTRACT}"$'\n'
        echo "$CONTRIBUTORS"$'\n'
        echo $'## Chapters\n\n'"$CHAPTERS"$'\n'
        echo "$INTRODUCTION"$'\n'
        echo $'## Discussion\n\n'"$PASSAGES"$'\n'
        echo "$CONCLUSION"$'\n'
        if [ "$include_refs" = true ]; then
            echo $'## References\n\n'"$REFERENCES"$'\n'
        fi
    ) | sed -E 's/\[\[([0-9]+)\]\]\([^)]+\)//g' | sed -E 's/\[([0-9]+)\]//g'
}

# Generate documents
echo "Generating Documents..."
sanitized_title=$(echo "$TITLE" | sed 's/[^a-zA-Z0-9]/_/g')

# Generate without references
generate_document false "-no-refs" | tee \
    >(pandoc -o "${sanitized_title}-no-refs.md" --from markdown --to markdown) \
    >(pandoc -o "${sanitized_title}-no-refs.pdf" --from markdown --pdf-engine=xelatex) >/dev/null

# Generate with references
generate_document true "-refs" | \
    pandoc -o "${sanitized_title}-refs.pdf" --from markdown+header_attributes --pdf-engine=xelatex

# Handle image extraction
if [ "$IMAGES" = true ]; then
    echo "Extracting images..."
    IMAGE_OUTPUT=$(plato --images "$URL" --lang "$LANG" $MODEL_FLAG | sed '/^$/d' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}')
    
    if [ -n "$IMAGE_OUTPUT" ]; then
        TMP_IMG_DIR=$(mktemp -d)
        
        echo "$IMAGE_OUTPUT" | while read -r img_path; do
            if [ -f ".platogram-cache/$img_path" ]; then
                cp ".platogram-cache/$img_path" "$TMP_IMG_DIR/"
            else
                echo "Warning: Image file not found: $img_path"
            fi
        done
        
        ZIP_FILE="${sanitized_title}-images.zip"
        if [ "$(ls -A "$TMP_IMG_DIR")" ]; then
            zip -j "$ZIP_FILE" "$TMP_IMG_DIR"/* && echo "Images saved to $ZIP_FILE"
        else
            echo "No valid images found to archive"
        fi
        
        rm -rf "$TMP_IMG_DIR"
    else
        echo "No images found or extracted."
    fi
fi

# Output verbose information if requested
if [ "$VERBOSE" = true ]; then
    echo "<title>"
    echo "$TITLE"
    echo "</title>"
    echo
    echo "<abstract>"
    echo "$ABSTRACT"
    echo "</abstract>"
fi