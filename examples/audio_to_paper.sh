#!/bin/bash

set -e

# Constants
MAX_RETRIES=3
RETRY_DELAY=2
STATUS_TIMEOUT=300  # 5 minutes timeout

DEBUG_LOG() {
    echo "[DEBUG] $*" >&2
}

# Function to handle errors
handle_error() {
    local exit_code=$?
    echo "Error occurred in script at line $1, exit code: $exit_code"
    exit $exit_code
}

trap 'handle_error ${LINENO}' ERR

# Function to check status with timeout
check_status_with_timeout() {
    local start_time=$(date +%s)
    
    while true; do
        if plato --status "$URL" $MODEL_FLAG | grep -q "complete"; then
            return 0
        fi
        
        current_time=$(date +%s)
        if ((current_time - start_time > STATUS_TIMEOUT)); then
            echo "Error: Status check timed out after $STATUS_TIMEOUT seconds"
            return 1
        fi
        
        sleep 5
    done
}

# Function for retrying commands with exponential backoff and content validation
get_with_retry() {
    local cmd="$1"
    local error_msg="$2"
    local content_type="$3"
    local retries=0
    local output=""
    local max_wait=30
    
    while ((retries < MAX_RETRIES)); do
        # Check if content processing is complete
        if ! check_status_with_timeout; then
            echo "Content processing check failed" >&2
            return 1
        fi
        
        DEBUG_LOG "Executing command: $cmd"
        output=$(eval "$cmd" || true)
        DEBUG_LOG "Command output for $content_type: $output"
        
        # Content-specific validation
        if [ -n "$output" ]; then
            case "$content_type" in
                "title")
                    if [ "$output" != "Missing Title" ]; then
                        echo "$output"
                        return 0
                    fi
                    ;;
                "abstract")
                    if [ "$output" != "Missing Summary" ]; then
                        echo "$output"
                        return 0
                    fi
                    ;;
                *)
                    echo "$output"
                    return 0
                    ;;
            esac
        fi
        
        ((retries++))
        echo "Attempt $retries failed for $content_type, retrying in $((RETRY_DELAY ** retries)) seconds..." >&2
        sleep $((min(RETRY_DELAY ** retries, max_wait)))
    done
    
    # Attempt generation fallback
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
MODEL="gemini"

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

# Configure model-specific settings
# Constants for status checking
STATUS_TIMEOUT=300  # 5 minutes timeout
MAX_STATUS_RETRIES=30
INITIAL_RETRY_DELAY=2

# Function to check processing status with timeout
check_status_with_timeout() {
    local start_time=$(date +%s)
    local status_checked=false
    local retry_count=0
    
    while ! $status_checked; do
        if plato --status "$URL" $MODEL_FLAG 2>/dev/null | grep -q "complete"; then
            status_checked=true
            return 0
        fi
        
        current_time=$(date +%s)
        if ((current_time - start_time > STATUS_TIMEOUT)); then
            echo "Error: Status check timed out after $STATUS_TIMEOUT seconds"
            return 1
        fi
        
        ((retry_count++))
        sleep $INITIAL_RETRY_DELAY
    done
}

# Model configuration and validation
case "$MODEL" in
"gemini")
    if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
        echo "Error: GOOGLE_CLOUD_PROJECT environment variable not set"
        exit 1
    fi
    if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
        exit 1
    fi
    MODEL_FLAG="--model gemini"
    # Configure status checking for Gemini
    STATUS_TIMEOUT=300
    MAX_STATUS_RETRIES=30
    INITIAL_RETRY_DELAY=2
    ;;
"anthropic")
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "Error: ANTHROPIC_API_KEY not set"
        exit 1
    fi
    MODEL_FLAG="--anthropic-api-key $ANTHROPIC_API_KEY"
    # Configure status checking for Anthropic
    STATUS_TIMEOUT=600  # Longer timeout for Anthropic
    MAX_STATUS_RETRIES=40
    INITIAL_RETRY_DELAY=3
    ;;
*)
    echo "Error: Unsupported model: $MODEL"
    exit 1
    ;;
esac

# Log model configuration
DEBUG_LOG "Model configuration:"
DEBUG_LOG "  Model type: $MODEL"
DEBUG_LOG "  Status timeout: ${STATUS_TIMEOUT}s"
DEBUG_LOG "  Max retries: $MAX_STATUS_RETRIES"
DEBUG_LOG "  Initial delay: ${INITIAL_RETRY_DELAY}s"
# Log configuration
{
    DEBUG_LOG "========= Configuration ========="
    DEBUG_LOG "Model type: $MODEL"
    DEBUG_LOG "Project ID: $GOOGLE_CLOUD_PROJECT"
    DEBUG_LOG "Credentials: $GOOGLE_APPLICATION_CREDENTIALS"
    DEBUG_LOG "Language: $LANG"
    DEBUG_LOG "Images enabled: $IMAGES"
    DEBUG_LOG "=============================="
}

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
    # Spanish prompts remain unchanged
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
    if ! plato "$URL" ${IMAGES:+--images} --lang "$LANG" $MODEL_FLAG >/dev/null; then
        echo "Error: Failed to retrieve text from URL"
        exit 1
    fi
else
    echo "Transcribing audio to text using AssemblyAI..."
    if ! plato "$URL" ${IMAGES:+--images} --assemblyai-api-key "$ASSEMBLYAI_API_KEY" --lang "$LANG" $MODEL_FLAG >/dev/null; then
        echo "Error: Failed to transcribe audio"
        exit 1
    fi
fi

echo "Checking content processing status..."
if ! check_status_with_timeout; then
    echo "Error: Content processing timed out"
    exit 1
fi

echo "Fetching content..."

# Get content with retries and sanitization
DEBUG_LOG "Retrieving title..."
TITLE=$(get_with_retry "plato --title '$URL' --lang '$LANG' $MODEL_FLAG" "Generated Title" "title")
DEBUG_LOG "Retrieved title: $TITLE"

DEBUG_LOG "Retrieving abstract..."
ABSTRACT=$(get_with_retry "plato --abstract '$URL' --lang '$LANG' $MODEL_FLAG" "Generated Summary" "abstract")
DEBUG_LOG "Retrieved abstract: $ABSTRACT"

DEBUG_LOG "Retrieving passages..."
PASSAGES=$(get_with_retry "plato --passages --chapters --inline-references '$URL' --lang '$LANG' $MODEL_FLAG" "No content available" "passages")

DEBUG_LOG "Retrieving references..."
REFERENCES=$(get_with_retry "plato --references '$URL' --lang '$LANG' $MODEL_FLAG" "No references available" "references")

DEBUG_LOG "Retrieving chapters..."
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