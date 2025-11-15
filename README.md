## Subtitle Translator

Translate entire shows or movies in minutes with an OpenAI-powered CLI. The tool groups
`.srt` lines into context-aware batches (`[context_before] [lines] [context_after]`),
feeds them to Chat Completions, and resumes from partially translated files so you never
redo work.

### Features
- Context-rich prompts keep terminology consistent (y/z strings before/after each batch).
- Smart resume & autosave: flushes after every batch and skips `.tr.srt` files entirely.
- Folder mode mirrors your directory tree and renames `Show.en.srt → Show.tr.srt`
  automatically (or whatever matches the target language code).
- Configurable batch size, context window, sampling, and model choice.
- `.env` support for `OPENAI_API_KEY`; no secrets on the command line.

### Requirements
- [uv](https://docs.astral.sh/uv/) (for dependency management)
- Python 3.9+
- An OpenAI API key (either exported or stored in `.env`)

### Quick start
```bash
uv sync
echo "OPENAI_API_KEY=sk-your-key" > .env
uv run ./main.py example/sample.en.srt --target-language Turkish
```

Translate a whole library (recursively) with your saved defaults:
```bash
uv run ./main.py /path/to/subtitles --config config.json
```

### CLI highlights
- `--model` chooses the OpenAI model (`gpt-4o-mini` default).
- `--context-before/--context-after` tune how many neighbor lines guide each batch.
- `--batch-size` controls how many entries go in one API request.
- `--temp` / `--top_p` adjust sampling behavior.
- `--output` overrides the auto-generated target filename (must be a directory in folder
  mode). Without it, `*.en.srt` becomes `*.tr.srt`, `*.de.srt`, etc.
- Existing translations are detected: previously translated lines remain untouched.

### Configuration file
Store your preferred knobs in `config.json` so runs stay consistent:
```json
{
  "input_language": "English",
  "target_language": "Turkish",
  "model": "gpt-4o-mini",
  "context_before": 2,
  "context_after": 1,
  "batch_size": 4,
  "temp": 0.3,
  "top_p": 0.9,
  "output": "example/custom-output.srt"
}
```

Launch with:
```bash
uv run ./main.py example/ --config config.json
```
Flags still take precedence, and any missing values fall back to interactive prompts.

### Sample assets
- `example/sample.en.srt`: eight short cues for smoke testing.
- `config.json`: ready-to-tweak template for your favorite settings.

### How it works
1. Parse the source `.srt` into structured entries.
2. Batch lines (default = 3) with context before/after; include already translated lines
   from earlier batches so tone carries forward.
3. Call OpenAI’s Chat Completions API, requesting strict JSON for deterministic parsing.
4. Flush the updated `.srt` after every batch so you can quit/resume anytime.

Happy translating! Contributions, issues, and feature requests are welcome.
