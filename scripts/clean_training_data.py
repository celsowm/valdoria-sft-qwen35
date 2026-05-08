#!/usr/bin/env python3
"""Clean training data by removing structured formats.

Removes patterns like:
- tipo/campo/resposta
- decisão/motivo/ação_final/prioridade
- termo/resposta in roleplay
- Tags like ⟦VALDORIA-CANON-v*⟧, ⟦VALDORIA-RPG-v*⟧, etc.
"""

import json
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Remove structured patterns from text."""
    if not text:
        return text

    # Remove tags like ⟦VALDORIA-CANON-v3.1⟧, ⟦VALDORIA-RPG-v3.2⟧, etc.
    text = re.sub(r'⟦VALDORIA-[A-Z]+-v[\d.]+\⟧', '', text)

    # Remove structured patterns at the beginning of response
    # Pattern: tipo: ... campo: ... resposta: ...
    if 'tipo:' in text and 'resposta:' in text:
        # Extract just the response
        match = re.search(r'resposta:\s*(.+)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Pattern: decisão: ... motivo: ... ação_final: ... prioridade: ...
    if 'decisão:' in text and 'motivo:' in text:
        # Extract just the motivation (most natural part)
        match = re.search(r'motivo:\s*(.+?)(?=ação_final:|prioridade:|$)', text, re.DOTALL)
        if match:
            text = f"Decisão: {text.split('motivo:')[0].split('Decisão:')[-1].strip()}. Motivo: {match.group(1).strip()}"
        else:
            # Just remove structured fields
            text = re.sub(r'(decisão:|motivo:|ação_final:|prioridade:)\s*\w+', '', text).strip()

    # Pattern: termo: ... resposta: ... (roleplay)
    if 'termo:' in text and 'resposta:' in text:
        match = re.search(r'resposta:\s*(.+)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def clean_dataset(input_path: str, output_path: str):
    """Clean a JSONL dataset file."""
    input_p = Path(input_path)
    output_p = Path(output_path)

    cleaned = []
    modified = 0

    with open(input_p, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                messages = data.get('messages', [])

                for msg in messages:
                    if msg.get('role') == 'assistant':
                        original = msg['content']
                        cleaned_content = clean_text(original)
                        if cleaned_content != original:
                            modified += 1
                        msg['content'] = cleaned_content

                cleaned.append(data)
            except json.JSONDecodeError as e:
                print(f"Error on line {line_num}: {e}")
                continue

    with open(output_p, 'w', encoding='utf-8') as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Processed {len(cleaned)} examples")
    print(f"Modified {modified} assistant messages")
    print(f"Output: {output_p}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python clean_training_data.py <input.jsonl> [output.jsonl]")
        print("If output is not specified, will append .cleaned to input name")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.jsonl', '.cleaned.jsonl')

    clean_dataset(input_file, output_file)
