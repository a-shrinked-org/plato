import argparse
import os
import re
import sys
from pathlib import Path
from typing import Callable, Literal, Sequence
from urllib.parse import urlparse

from tqdm import tqdm

import platogram as plato
import platogram.ingest as ingest
from platogram.library import Library
from platogram.types import Assistant, Content, User
from platogram.utils import make_filesystem_safe

CACHE_DIR = Path("./.platogram-cache")


def format_time(ms):
    seconds = ms // 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_reference(url: str, transcript: list[plato.SpeechEvent], i: int) -> str:
    link = f" [[{i+1}]](#ts-{i + 1})"
    return link
    
def get_chapter(passage_marker: int, chapter_markers: list[int]) -> int | None:
    for start, end in zip(chapter_markers[:-1], chapter_markers[1:]):
        if start <= passage_marker < end:
            return start
    if passage_marker >= chapter_markers[-1]:
        return chapter_markers[-1]
    return None

def render_transcript(first, last, transcript, url):
    return "\n".join(
        [
            f"\n##### {{#ts-{i + 1}}}\n{i-first+1}. [{format_time(event.time_ms)}]({url}#t={event.time_ms // 1000}): {event.text}"
            for i, event in enumerate(transcript)
            if first <= i <= last
        ]
    )


def render_paragraph(p: str, render_reference_fn: Callable[[int], str]) -> str:
    references = sorted([int(i) for i in re.findall(r"【(\d+)】", p)])
    if not references:
        return p

    paragraph = re.sub(
        r"【(\d+)】",
        lambda match: render_reference_fn(int(match.group(1))),
        p,
    )

    return paragraph


def process_url(
    url_or_file: str,
    library: Library,
    anthropic_api_key: str | None = None,
    assemblyai_api_key: str | None = None,
    model_type: str = "gemini",  # Default to gemini
    extract_images: bool = False,
    lang: str | None = None,
) -> Content:
    """Process a URL or file."""
    print("=== Debug: Process URL Configuration ===")
    print(f"URL: {url_or_file}")
    print(f"Model: {model_type}")
    print(f"Project ID: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    print(f"Credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    print(f"Language: {lang}")
    print("===================================")

    # Configure model first to fail fast if credentials are missing
    if model_type == "gemini":
        if not os.getenv("GOOGLE_CLOUD_PROJECT") or not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError("Missing Gemini credentials")
    elif model_type == "anthropic" and not anthropic_api_key:
        raise ValueError("Missing Anthropic API key")
    
    # Initialize ASR if needed
    asr = None
    if assemblyai_api_key:
        print("Debug: Initializing ASR with AssemblyAI")
        import assemblyai as aai
        
        # Configure faster polling
        aai.settings.polling_interval = 1.0
        
        # Convert language code to AssemblyAI format
        aai_lang = "en" if lang == "en" else "es"
        
        # Configure transcriber with language
        config = aai.TranscriptionConfig(
            language_code=aai_lang
        )
        asr = aai.Transcriber(config=config)
        asr.api_key = assemblyai_api_key
    
    # Extract transcript
    print("Debug: Starting transcript extraction")
    try:
        # Remove lang parameter when calling transcribe
        if asr:
            transcript = plato.extract_transcript(url_or_file, asr)
        else:
            transcript = plato.extract_transcript(url_or_file, asr, lang=lang)
    except ValueError as e:
        if "No subtitles found and no ASR model provided" in str(e):
            print("Error: Audio file detected but no ASR key provided")
            raise
        raise
    
    # Initialize LLM only after successful transcript extraction
    print(f"Debug: Initializing LLM model: {model_type}")
    llm = plato.llm.get_model(
        f"{model_type}/{'gemini-2.0-flash-001' if model_type == 'gemini' else 'claude-3-5-sonnet'}", 
        anthropic_api_key if model_type == "anthropic" else None
    )
    
    # Process content
    content = plato.index(transcript, llm, lang=lang)
    
    if extract_images:
        print("Debug: Extracting images")
        images_dir = library.home / make_filesystem_safe(url_or_file)
        images_dir.mkdir(exist_ok=True)
        timestamps_ms = [event.time_ms for event in content.transcript]
        images = ingest.extract_images(url_or_file, images_dir, timestamps_ms)
        content.images = [str(image.relative_to(library.home)) for image in images]
    
    return content

def prompt_context(
        context: list[Content],
        prompt: Sequence[Assistant | User],
        context_size: Literal["small", "medium", "large"],
        model_type: str = "gemini",
        anthropic_api_key: str | None = None,
    ) -> str:
        model_name = f"{model_type}/{'gemini-2.0-flash-001' if model_type == 'gemini' else 'claude-3-5-sonnet'}"
        llm = plato.llm.get_model(model_name, anthropic_api_key if model_type == "anthropic" else None)
        
        response = llm.prompt(
            prompt=prompt,
            context=context,
            context_size=context_size,
        )
        return response

def is_uri(s):
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="Platogram CLI")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="URLs and files to query, if none provided, will use all content",
    )
    parser.add_argument("--lang", help="Content language: en, es")
    parser.add_argument(
        "--model",
        choices=["anthropic", "gemini"],
        default="gemini",
        help="Model to use for processing (default: gemini)"
    )
    parser.add_argument("--anthropic-api-key", help="Anthropic API key")
    parser.add_argument("--assemblyai-api-key", help="AssemblyAI API key (optional)")
    parser.add_argument("--retrieve", default=None, help="Number of results to retrieve")
    parser.add_argument("--generate", action="store_true", help="Generate content")
    parser.add_argument("--query", help="Query for retrieval and generation")
    parser.add_argument(
        "--context-size",
        choices=["small", "medium", "large"],
        default="small",
        help="Context size for prompting",
    )
    parser.add_argument("--title", action="store_true", help="Include title")
    parser.add_argument("--abstract", action="store_true", help="Include abstract")
    parser.add_argument("--passages", action="store_true", help="Include passages")
    parser.add_argument("--chapters", action="store_true", help="Include chapters")
    parser.add_argument("--references", action="store_true", help="Include references")
    parser.add_argument("--images", action="store_true", help="Include images")
    parser.add_argument("--origin", action="store_true", help="Include origin URL")
    parser.add_argument(
        "--retrieval-method",
        choices=["keyword", "semantic", "dumb"],
        default="dumb",
        help="Retrieval method",
    )
    parser.add_argument(
        "--prefill",
        default="",
        help="Nudge the model to continue the provided sentence",
    )
    parser.add_argument(
        "--inline-references", 
        action="store_true", 
        help="Render references inline"
    )
    
    args = parser.parse_args()
    
    # Set language
    lang = args.lang if args.lang else "en"
    
    # Get AssemblyAI key from environment if not provided in args
    if not args.assemblyai_api_key:
        args.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    if args.model == "gemini":
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            print("Error: GOOGLE_CLOUD_PROJECT environment variable not set")
            sys.exit(1)
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
            sys.exit(1)
    
    # Initialize library based on retrieval method
    if args.retrieval_method == "semantic":
        library = plato.library.get_semantic_local_chroma(CACHE_DIR)
    elif args.retrieval_method == "keyword":
        library = plato.library.get_keyword_local_bm25(CACHE_DIR)
    elif args.retrieval_method == "dumb":
        library = plato.library.get_local_dumb(CACHE_DIR)
    else:
        raise ValueError(f"Invalid retrieval method: {args.retrieval_method}")
    
    # Get content context
    if not args.inputs:
        ids = library.ls()
        context = [library.get_content(id) for id in ids]
    else:
        ids = [make_filesystem_safe(url_or_file) for url_or_file in args.inputs]
        context = [
            process_url(
                url_or_file,
                library,
                args.anthropic_api_key,
                args.assemblyai_api_key,
                args.model,
                extract_images=args.images,
                lang=lang,
            )
            for url_or_file in args.inputs
        ]
    
    # Handle keyword retrieval
    if args.retrieval_method == "keyword":
        library.put(ids[0], context[0])
    
    # Handle content retrieval
    if args.retrieve:
        n_results = int(args.retrieve)
        context, scores = library.retrieve(args.query, n_results, ids)
    
    # Initialize result string
    result = ""
    
    # Handle content generation
    if args.generate:
        if not args.query:
            raise ValueError("Query is required for generation")
        
        prompt = [User(content=args.query)]
        if args.prefill:
            prompt = [User(content=args.query), Assistant(content=args.prefill)]
        
        result += f"""\n\n{
            prompt_context(
                context=context,
                prompt=prompt,
                context_size=args.context_size,
                model_type=args.model,
                anthropic_api_key=args.anthropic_api_key
            )}\n\n"""
    
    # Process each content item
    for content in context:
        # Handle images
        if args.images and content.images:
            images = "\n".join([str(image) for image in content.images])
            result += f"""{images}\n\n\n\n"""
    
        # Handle origin URL
        if args.origin:
            result += f"""{content.origin}\n\n\n\n"""
    
        # Handle title
        if args.title:
            result += f"""{content.title}\n\n\n\n"""
    
        # Handle abstract
        if args.abstract:
            result += f"""{content.summary}\n\n\n\n"""
    
        # Handle passages and chapters
        if args.passages:
            passages = ""
            if args.chapters:
                current_chapter = None
                chapter_markers = list(content.chapters.keys())
                for passage in content.passages:
                    passage_markers = [int(m) for m in re.findall(r"【(\d+)】", passage)]
                    chapter_marker = get_chapter(passage_markers[0], chapter_markers) if passage_markers else None
                    if chapter_marker is not None and chapter_marker != current_chapter:
                        passages += f"### {content.chapters[chapter_marker]}\n\n"
                        current_chapter = chapter_marker
                    passages += f"{passage.strip()}\n\n"
            else:
                passages = "\n\n".join(
                    passage.strip() for passage in content.passages
                )
            result += f"""{passages}\n\n\n\n"""
    
        # Handle chapters without passages
        if args.chapters and not args.passages:
            chapters = "\n".join(
                f"- {chapter} [{i}]" for i, chapter in content.chapters.items()
            )
            result += f"""{chapters}\n\n\n\n"""
    
        # Handle references
        if args.references:
            result += f"""{render_transcript(0, len(content.transcript), content.transcript, content.origin)}\n\n\n\n"""
    
        # Handle inline references
        if args.inline_references:
            render_reference_fn = lambda i: render_reference(
                content.origin or "", content.transcript, i
            )
        else:
            render_reference_fn = lambda _: ""
    
        result = render_paragraph(result, render_reference_fn)

    print(result)


if __name__ == "__main__":
    main()
