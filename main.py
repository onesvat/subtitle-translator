#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:  # Ensure Python 3.9 stdlib exposes packages_distributions
    import importlib.metadata as _stdlib_metadata

    if not hasattr(_stdlib_metadata, "packages_distributions"):
        try:
            import importlib_metadata as _backport_metadata
        except ImportError:  # pragma: no cover - dependency missing
            _backport_metadata = None
        if _backport_metadata and hasattr(_backport_metadata, "packages_distributions"):
            _stdlib_metadata.packages_distributions = _backport_metadata.packages_distributions
except Exception:  # pragma: no cover - non-critical defensive patch
    pass

DEFAULT_BATCH_SIZE = 3
DEFAULT_CONTEXT = 1
DEFAULT_MODEL = "gpt-4o-mini"
LANGUAGE_CODE_MAP = {
    "english": "en",
    "turkish": "tr",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "hindi": "hi",
    "arabic": "ar",
}
CONFIG_FIELDS = [
    "input_language",
    "target_language",
    "model",
    "context_before",
    "context_after",
    "batch_size",
    "temp",
    "top_p",
]


@dataclass(frozen=True)
class SrtEntry:
    index: int
    start_ts: str
    end_ts: str
    text: str


@dataclass
class RuntimeConfig:
    input_language: str
    target_language: str
    model: str
    context_before: int
    context_after: int
    batch_size: int
    temperature: float | None
    top_p: float | None


class ProviderClient:
    def __init__(
        self,
        model: str,
        *,
        temperature: float | None,
        top_p: float | None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self._openai_client = None
        self._openai_lock = threading.Lock()

    def translate(self, system_prompt: str, user_prompt: str) -> str:
        return self._translate_openai(system_prompt, user_prompt)

    def _translate_openai(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        if OpenAI is None:
            raise RuntimeError("The openai package is not installed")
        if self._openai_client is None:
            with self._openai_lock:
                if self._openai_client is None:
                    self._openai_client = OpenAI(api_key=api_key)
        completion_args = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None:
            completion_args["temperature"] = self.temperature
        if self.top_p is not None:
            completion_args["top_p"] = self.top_p
        response = self._openai_client.chat.completions.create(**completion_args)
        return response.choices[0].message.content.strip()

def parse_srt(path: Path) -> List[SrtEntry]:
    text = path.read_text(encoding="utf-8")
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return [_block_to_entry(block) for block in blocks]


def _block_to_entry(block: List[str]) -> SrtEntry:
    if len(block) < 2:
        raise ValueError(f"Malformed SRT block: {block}")
    index = int(block[0].strip())
    times = block[1]
    if "-->" not in times:
        raise ValueError(f"Invalid time range '{times}' for block {index}")
    start, end = [x.strip() for x in times.split("-->")]
    text = "\n".join(block[2:]).strip() if len(block) > 2 else ""
    return SrtEntry(index=index, start_ts=start, end_ts=end, text=text)


def write_srt(entries: Sequence[SrtEntry], path: Path) -> None:
    lines: List[str] = []
    for entry in entries:
        lines.append(str(entry.index))
        lines.append(f"{entry.start_ts} --> {entry.end_ts}")
        lines.append(entry.text)
        lines.append("")
    contents = "\n".join(lines).strip() + "\n"
    path.write_text(contents, encoding="utf-8")


def build_system_prompt(input_language: str, target_language: str) -> str:
    return (
        "You are a meticulous subtitle translator. "
        f"Convert {input_language} subtitles into {target_language} while respecting idioms, "
        "tone, punctuation, and speaker markers. Keep original line breaks, numbers, and "
        "timestamps, and never omit or hallucinate content."
    )


def build_user_prompt(
    chunk: Sequence[SrtEntry],
    context_before: Sequence[SrtEntry],
    context_after: Sequence[SrtEntry],
    *,
    target_language: str,
    translated_context: Sequence[tuple[int, str]] | None = None,
) -> str:
    template = (
        "Translate the highlighted lines into {lang}. Use the context-only sections "
        "to understand tone. A translated-context section may appear; treat it as "
        "reference only and do not modify those strings. Consider the focus entries a "
        "single paragraph: you may redistribute words or clauses between the entries "
        "to honor natural {lang} word order, as long as each entry remains aligned with "
        "its timestamp and the overall meaning stays intact. Respond with strict JSON "
        'in the form {{"translations":[{{"index":<int>,"text":"<translated line>"}}]}}.\n'
    )
    header = template.format(lang=target_language)
    context_section = _format_entries("CONTEXT BEFORE", context_before)
    translated_section = _format_translated_context(translated_context)
    chunk_section = _format_entries("LINES TO TRANSLATE", chunk)
    after_section = _format_entries("CONTEXT AFTER", context_after)
    ordering = (
        "[context_before]\n{before}\n\n"
        "[context_before_translated]\n{translated}\n\n"
        "[lines]\n{chunk}\n\n"
        "[context_after]\n{after}"
    ).format(
        before=context_section,
        translated=translated_section,
        chunk=chunk_section,
        after=after_section,
    )
    return f"{header}\n{ordering}"


def _format_entries(title: str, entries: Sequence[SrtEntry]) -> str:
    if not entries:
        return f"{title}:\n(none)"
    formatted = "\n".join(f"{entry.index}: {entry.text}" for entry in entries)
    return f"{title}:\n{formatted}"


def _format_translated_context(translated: Sequence[tuple[int, str]] | None) -> str:
    if not translated:
        return "(none)"
    return "\n".join(f"{index}: {text}" for index, text in translated)


def _format_index_groups(entries: Sequence[SrtEntry]) -> str:
    if not entries:
        return "none"
    indices = [entry.index for entry in entries]
    groups: List[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    formatted: List[str] = []
    for begin, end in groups:
        if begin == end:
            formatted.append(f"#{begin}")
        else:
            formatted.append(f"#{begin}-{end}")
    return ", ".join(formatted)


def _summarize_text(text: str, limit: int = 80) -> str:
    compact = text.replace("\n", " / ").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def iter_chunks(
    entries: Sequence[SrtEntry],
    *,
    batch_size: int,
    context_before: int,
    context_after: int,
) -> Iterable[Tuple[Sequence[SrtEntry], Sequence[SrtEntry], Sequence[SrtEntry]]]:
    total = len(entries)
    for start in range(0, total, batch_size):
        chunk = entries[start : start + batch_size]
        before_start = max(0, start - context_before)
        before = entries[before_start:start]
        after = entries[start + batch_size : start + batch_size + context_after]
        yield chunk, before, after


def parse_translation_blob(blob: str) -> Dict[int, str]:
    json_blob = _extract_json_object(blob.strip())
    try:
        data = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        preview = json_blob[:500]
        raise ValueError(
            "Model response was not valid JSON. "
            f"{exc.msg} at char {exc.pos}. Partial payload:\n{preview}"
        ) from exc
    translations = data.get("translations")
    if not translations:
        raise ValueError("Translation payload missing 'translations'")
    result: Dict[int, str] = {}
    for item in translations:
        index = int(item["index"])
        text = str(item["text"]).strip()
        result[index] = text
    return result


def _extract_json_object(blob: str) -> str:
    start = blob.find("{")
    end = blob.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Response did not contain a JSON object")
    return blob[start : end + 1]


def load_config_data(config_path: Path | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file {config_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a JSON object")
    return data


def apply_config_defaults(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    if not config:
        return
    for field in CONFIG_FIELDS:
        if getattr(args, field, None) is None and field in config:
            setattr(args, field, config[field])
    if getattr(args, "output", None) is None and config.get("output"):
        args.output = Path(config["output"])


def load_existing_translations(
    source_entries: Sequence[SrtEntry],
    output_path: Path,
) -> Dict[int, str]:
    if not output_path.exists():
        return {}
    try:
        saved_entries = parse_srt(output_path)
    except Exception as exc:
        print(f"Warning: could not parse existing translation file {output_path}: {exc}")
        return {}
    if len(saved_entries) != len(source_entries):
        print(
            "Warning: existing translation file entry count does not match source. "
            "Ignoring previously saved translations."
        )
        return {}
    translations: Dict[int, str] = {}
    for source, saved in zip(source_entries, saved_entries):
        if source.index != saved.index:
            print(
                "Warning: existing translation file indices do not match source. "
                "Ignoring previously saved translations."
            )
            return {}
        if saved.text and saved.text != source.text:
            translations[source.index] = saved.text
    if translations:
        print(f"Loaded {len(translations)} existing translations from {output_path}")
    return translations


def _materialize_entries(
    entries: Sequence[SrtEntry],
    translations: Dict[int, str],
) -> List[SrtEntry]:
    result: List[SrtEntry] = []
    for entry in entries:
        result.append(
            SrtEntry(
                index=entry.index,
                start_ts=entry.start_ts,
                end_ts=entry.end_ts,
                text=translations.get(entry.index, entry.text),
            )
        )
    return result


def guess_language_code(language: str | None) -> str | None:
    if not language:
        return None
    key = language.strip().lower()
    if not key:
        return None
    if key in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[key]
    compact = "".join(ch for ch in key if ch.isalpha())
    if len(compact) >= 2:
        return compact[:2]
    return None


def derive_output_name(source_path: Path, target_language: str) -> str:
    lang_code = guess_language_code(target_language)
    suffixes = source_path.suffixes
    if lang_code and suffixes and suffixes[-1].lower() == ".srt":
        base_name = source_path.name
        if len(suffixes) >= 2 and len(suffixes[-2]) == 3 and suffixes[-2].startswith("."):
            trim_len = len("".join(suffixes[-2:]))
            prefix = base_name[: -trim_len]
            return f"{prefix}.{lang_code}.srt"
        else:
            prefix = base_name[: -len(suffixes[-1])]
            return f"{prefix}.{lang_code}.srt"
    return source_path.with_suffix(".translated.srt").name


def is_target_language_file(path: Path, target_code: str | None) -> bool:
    if not target_code:
        return False
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-1].lower() == ".srt":
        return suffixes[-2].lstrip(".").lower() == target_code.lower()
    return False


def translate_entries(
    entries: Sequence[SrtEntry],
    *,
    provider: ProviderClient,
    config: RuntimeConfig,
    existing_translations: Dict[int, str],
    output_path: Path,
) -> List[SrtEntry]:
    if not entries:
        return []
    system_prompt = build_system_prompt(config.input_language, config.target_language)
    jobs = list(
        iter_chunks(
            entries,
            batch_size=config.batch_size,
            context_before=config.context_before,
            context_after=config.context_after,
        )
    )
    translations: Dict[int, str] = dict(existing_translations)
    if translations:
        current_entries = _materialize_entries(entries, translations)
        write_srt(current_entries, output_path)
    for chunk, before, after in jobs:
        chunk_to_translate = [entry for entry in chunk if entry.index not in translations]
        chunk_label = _format_index_groups(chunk_to_translate or chunk)
        before_label = _format_index_groups(before)
        after_label = _format_index_groups(after)
        if not chunk_to_translate:
            print(
                f"Skipping {chunk_label} "
                f"(context before: {before_label}; context after: {after_label})"
            )
            continue
        # print(
        #     f"Starting translation for {chunk_label} "
        #     f"(context before: {before_label}; context after: {after_label})"
        # )
        translated_context = [
            (entry.index, translations[entry.index])
            for entry in before
            if entry.index in translations
        ]
        prompt = build_user_prompt(
            chunk_to_translate,
            before,
            after,
            target_language=config.target_language,
            translated_context=translated_context,
        )
        attempts = 3
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            raw = provider.translate(system_prompt, prompt)
            try:
                batch = parse_translation_blob(raw)
                last_error = None
                break
            except ValueError as exc:
                last_error = exc
                print(
                    f"Warning: {exc}. Attempt {attempt} of {attempts} for {chunk_label}."
                )
                if attempt < attempts:
                    prompt += (
                        "\n\nREMINDER: Return strictly valid JSON matching the schema. "
                        "Do not include trailing commas or stray quotes."
                    )
        if last_error is not None:
            raise last_error
        missing_indexes = {entry.index for entry in chunk_to_translate} - set(batch)
        for missing in missing_indexes:
            original = next(entry.text for entry in chunk_to_translate if entry.index == missing)
            batch[missing] = original
        for entry in chunk_to_translate:
            translated_text = batch[entry.index]
            print(
                f'Translated #{entry.index} '
                f'"{_summarize_text(entry.text)}" -> "{_summarize_text(translated_text)}"'
            )
        translations.update(batch)
        current_entries = _materialize_entries(entries, translations)
        write_srt(current_entries, output_path)

    return _materialize_entries(entries, translations)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles via OpenAI with optional interactive prompts.",
    )
    parser.add_argument("srt_path", type=Path, help="Path to the source .srt file")
    parser.add_argument("--config", type=Path, help="Path to a JSON config file with default options")
    parser.add_argument("--output", type=Path, help="Destination .srt path (defaults to *.translated.srt)")
    parser.add_argument("--input-language", help="Language of the subtitle input (asks if omitted)")
    parser.add_argument("--target-language", help="Target translation language (asks if omitted)")
    parser.add_argument("--model", help="OpenAI model identifier to use")
    parser.add_argument("--context-before", type=int, help="How many Y strings precede each batch")
    parser.add_argument("--context-after", type=int, help="How many Z strings follow each batch")
    parser.add_argument("--batch-size", type=int, help="How many lines are translated per call")
    parser.add_argument("--temp", type=float, help="Sampling temperature (OpenAI default when omitted)")
    parser.add_argument("--top_p", type=float, help="Top-p nucleus sampling (OpenAI default when omitted)")
    return parser


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    input_language = _prompt_string(
        args.input_language,
        "Input language",
        default="English",
    )
    target_language = _prompt_string(
        args.target_language,
        "Target language",
        default="Turkish",
    )
    model = _prompt_string(
        args.model,
        "OpenAI model",
        default=DEFAULT_MODEL,
    )
    context_before = _prompt_int(
        args.context_before,
        "Context before (Y strings)",
        default=DEFAULT_CONTEXT,
        minimum=0,
    )
    context_after = _prompt_int(
        args.context_after,
        "Context after (Z strings)",
        default=DEFAULT_CONTEXT,
        minimum=0,
    )
    batch_size = _prompt_int(
        args.batch_size,
        "Batch size (lines translated together)",
        default=DEFAULT_BATCH_SIZE,
        minimum=1,
    )
    return RuntimeConfig(
        input_language=input_language,
        target_language=target_language,
        model=model,
        context_before=context_before,
        context_after=context_after,
        batch_size=batch_size,
        temperature=args.temp,
        top_p=args.top_p,
    )


def _prompt_string(current: str | None, label: str, *, default: str) -> str:
    if current:
        return current
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        return raw


def _prompt_int(current: int | None, label: str, *, default: int, minimum: int) -> int:
    if current is not None:
        if current < minimum:
            raise ValueError(f"{label} must be >= {minimum}")
        return current
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        try:
            value = default if not raw else int(raw)
        except ValueError:
            print("Enter a valid integer.")
            continue
        if value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        return value


def process_single_file(
    srt_file: Path,
    provider: ProviderClient,
    config: RuntimeConfig,
    output_path: Path,
) -> None:
    print(f"\n=== Translating {srt_file} ===")
    srt_entries = parse_srt(srt_file)
    existing_translations = load_existing_translations(srt_entries, output_path)
    if len(existing_translations) == len(srt_entries):
        print(f"Existing translation at {output_path} already covers all entries.")
        return
    translated = translate_entries(
        srt_entries,
        provider=provider,
        config=config,
        existing_translations=existing_translations,
        output_path=output_path,
    )
    write_srt(translated, output_path)
    print(f"Wrote translated subtitles to {output_path}")


def main() -> None:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()
    config_data = load_config_data(args.config)
    apply_config_defaults(args, config_data)
    if not args.srt_path.exists():
        raise FileNotFoundError(f"SRT path not found: {args.srt_path}")
    config = resolve_runtime_config(args)
    target_code = guess_language_code(config.target_language)
    provider = ProviderClient(
        config.model,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    if args.srt_path.is_dir():
        if args.output and not args.output.is_dir():
            raise ValueError("--output must point to a directory when translating a folder")
        base_input = args.srt_path
        base_output = args.output or base_input
        srt_files = sorted(
            path
            for path in base_input.rglob("*.srt")
            if not path.name.endswith(".translated.srt")
        )
        if not srt_files:
            print(f"No .srt files found under {base_input}")
            return
        for srt_file in srt_files:
            if is_target_language_file(srt_file, target_code):
                print(f"Skipping {srt_file} (already target language)")
                continue
            relative = srt_file.relative_to(base_input)
            output_name = derive_output_name(srt_file, config.target_language)
            target_path = (base_output / relative).with_name(output_name)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            process_single_file(srt_file, provider, config, target_path)
    else:
        if is_target_language_file(args.srt_path, target_code):
            print(f"Input {args.srt_path} already matches target language; nothing to do.")
            return
        if args.output:
            output_path = Path(args.output)
        else:
            output_name = derive_output_name(args.srt_path, config.target_language)
            output_path = args.srt_path.with_name(output_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        process_single_file(args.srt_path, provider, config, output_path)


if __name__ == "__main__":
    main()
