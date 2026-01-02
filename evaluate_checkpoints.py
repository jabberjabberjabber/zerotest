import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math
import argparse
from pathlib import Path
import json

# ============================================================================
# ROMAN NUMERAL UTILITIES
# ============================================================================

def int_to_roman_subtractive(num):
    """Standard Roman numerals with subtractive notation"""
    if num <= 0 or num > 3999:
        raise ValueError(f"Number {num} out of range")

    values = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]

    result = []
    for value, numeral in values:
        count = num // value
        if count:
            result.append(numeral * count)
            num -= value * count
    return ''.join(result)

def int_to_roman_additive(num):
    """Additive-only Roman numerals (VIIII instead of IX)"""
    if num <= 0 or num > 3999:
        raise ValueError(f"Number {num} out of range")

    values = [(1000, 'M'), (500, 'D'), (100, 'C'), (50, 'L'), (10, 'X'), (5, 'V'), (1, 'I')]

    result = []
    for value, numeral in values:
        count = num // value
        if count:
            result.append(numeral * count)
            num -= value * count
    return ''.join(result)

def roman_to_int(s):
    """Convert Roman numeral to integer (handles both notations)"""
    if s == 'N':
        return 0

    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev_value = 0

    for char in reversed(s):
        value = values[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total

def subtractive_to_additive(roman):
    """Convert subtractive notation to additive"""
    replacements = [
        ('CM', 'DCCCC'), ('CD', 'CCCC'),
        ('XC', 'LXXXX'), ('XL', 'XXXX'),
        ('IX', 'VIIII'), ('IV', 'IIII')
    ]

    for sub, add in replacements:
        roman = roman.replace(sub, add)
    return roman

def sort_roman(roman):
    """Sort Roman numeral letters from largest to smallest"""
    order = {'M': 0, 'D': 1, 'C': 2, 'L': 3, 'X': 4, 'V': 5, 'I': 6}
    return ''.join(sorted(roman, key=lambda x: order[x]))

def reduce_roman(roman):
    """Apply internal sums (IIIII->V, VV->X, etc.)"""
    changed = True
    while changed:
        changed = False
        old_roman = roman

        if 'IIIII' in roman:
            roman = roman.replace('IIIII', 'V', 1)
            changed = True
            continue
        if 'VV' in roman:
            roman = roman.replace('VV', 'X', 1)
            changed = True
            continue
        if 'XXXXX' in roman:
            roman = roman.replace('XXXXX', 'L', 1)
            changed = True
            continue
        if 'LL' in roman:
            roman = roman.replace('LL', 'C', 1)
            changed = True
            continue
        if 'CCCCC' in roman:
            roman = roman.replace('CCCCC', 'D', 1)
            changed = True
            continue
        if 'DD' in roman:
            roman = roman.replace('DD', 'M', 1)
            changed = True
            continue

    return roman

def additive_to_subtractive(roman):
    """Convert additive notation back to subtractive"""
    replacements = [
        ('DCCCC', 'CM'), ('CCCC', 'CD'),
        ('LXXXX', 'XC'), ('XXXX', 'XL'),
        ('VIIII', 'IX'), ('IIII', 'IV')
    ]

    for add, sub in replacements:
        if add in roman:
            roman = roman.replace(add, sub, 1)

    return roman

def subtract_roman(a_roman, b_roman):
    """Subtract b_roman from a_roman using the algorithm"""
    # Step 1: Convert to additive notation
    a_add = subtractive_to_additive(a_roman)
    b_add = subtractive_to_additive(b_roman)

    # Convert to list for easier manipulation
    a_chars = list(a_add)
    b_chars = list(b_add)

    # Step 2 & 3: Eliminate common symbols and expand as needed
    while b_chars:
        # Try to eliminate common symbols
        changed = False
        for char in b_chars[:]:  # Iterate over a copy
            if char in a_chars:
                a_chars.remove(char)
                b_chars.remove(char)
                changed = True
                break

        if changed:
            continue

        # No common symbols, need to expand
        if not b_chars:
            break

        # Find the largest remaining symbol in B
        b_largest = b_chars[0]

        # Find a larger symbol in A to expand
        expansion_map = {
            'M': 'DD',
            'D': 'CCCCC',
            'C': 'LL',
            'L': 'XXXXX',
            'X': 'VV',
            'V': 'IIIII'
        }

        # Find first symbol in A larger than b_largest
        order = {'M': 0, 'D': 1, 'C': 2, 'L': 3, 'X': 4, 'V': 5, 'I': 6}
        b_order = order[b_largest]

        expanded = False
        for i, a_char in enumerate(a_chars):
            if order[a_char] < b_order:  # Larger symbol (smaller order number)
                # Expand this symbol
                if a_char in expansion_map:
                    expansion = expansion_map[a_char]
                    a_chars[i:i+1] = list(expansion)
                    expanded = True
                    break

        if not expanded:
            # Can't expand anymore - this shouldn't happen with valid inputs
            break

    # Step 4: Convert result back to subtractive notation
    result = ''.join(a_chars)
    result = sort_roman(result)
    result = reduce_roman(result)
    result = additive_to_subtractive(result)

    return result

# ============================================================================
# TOKENIZERS
# ============================================================================

class CharTokenizer:
    """Tokenizer for algorithm-based trainer"""
    def __init__(self):
        # Static vocabulary
        vocab = ['<PAD>', '<EOS>', 'I', 'V', 'X', 'L', 'C', 'D', 'M', 'N', '+', '-', '=', '|', ' ']

        self.pad_token = '<PAD>'
        self.eos_token = '<EOS>'

        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        self.vocab_size = len(vocab)

        self.pad_idx = self.char_to_idx[self.pad_token]
        self.eos_idx = self.char_to_idx[self.eos_token]

    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            if text[i:i+len(self.eos_token)] == self.eos_token:
                tokens.append(self.eos_idx)
                i += len(self.eos_token)
            elif text[i:i+len(self.pad_token)] == self.pad_token:
                tokens.append(self.pad_idx)
                i += len(self.pad_token)
            else:
                if text[i] not in self.char_to_idx:
                    raise ValueError(f"Character '{text[i]}' not in vocabulary")
                tokens.append(self.char_to_idx[text[i]])
                i += 1
        return tokens

    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices if i != self.pad_idx])

class RomanTokenizer:
    """Tokenizer for math-based trainer"""
    def __init__(self):
        chars = ['I', 'V', 'X', 'L', 'C', 'D', 'M', 'N',
                 '+', '-', '*', '/', '=', ' ']

        self.pad_token = '<PAD>'
        self.eos_token = '<EOS>'
        chars.extend([self.pad_token, self.eos_token])

        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        self.pad_idx = self.char_to_idx[self.pad_token]
        self.eos_idx = self.char_to_idx[self.eos_token]

    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            if text[i:i+len(self.eos_token)] == self.eos_token:
                tokens.append(self.eos_idx)
                i += len(self.eos_token)
            elif text[i:i+len(self.pad_token)] == self.pad_token:
                tokens.append(self.pad_idx)
                i += len(self.pad_token)
            else:
                tokens.append(self.char_to_idx[text[i]])
                i += 1
        return tokens

    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices if idx != self.pad_idx])

# ============================================================================
# MODEL
# ============================================================================

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, max_seq_length=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.dropout(x)

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)

        logits = self.output(x)
        return logits

# ============================================================================
# GENERATION
# ============================================================================

def generate_text(model, prompt, tokenizer, device, max_new_tokens=200):
    """Generate text from a prompt"""
    model.eval()

    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Check if we've reached max sequence length
            if input_tensor.size(1) >= 127:  # Leave room for one more token
                break

            logits = model(input_tensor)
            next_token = logits[0, -1, :].argmax().item()

            if next_token == tokenizer.eos_idx:
                break

            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], dtype=torch.long).to(device)
            ], dim=1)

    return tokenizer.decode(input_tensor[0].cpu().tolist())

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def extract_final_answer(generated_text):
    """Extract the final answer from generated text"""
    # For ultra simple format: "A + B [intermediate] = C"
    if '=' in generated_text:
        parts = generated_text.split('=')
        if len(parts) >= 2:
            answer = parts[-1].strip()
            # Clean up EOS tokens and extra characters
            answer = answer.replace('<EOS>', '').strip()
            return answer

    return generated_text.strip()

def evaluate_addition_problems(model, tokenizer, device, num_problems=100,
                               min_val=1, max_val=50, use_additive=False):
    """Evaluate model on addition problems"""
    correct = 0
    total = 0
    errors = []

    for _ in range(num_problems):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

        if use_additive:
            a_roman = int_to_roman_additive(a)
            b_roman = int_to_roman_additive(b)
            expected_roman = int_to_roman_additive(a + b)
        else:
            a_roman = int_to_roman_subtractive(a)
            b_roman = int_to_roman_subtractive(b)
            expected_roman = int_to_roman_subtractive(a + b)

        prompt = f"{a_roman} + {b_roman} |"
        generated = generate_text(model, prompt, tokenizer, device)

        # Extract final answer
        answer = extract_final_answer(generated)

        # Check if answer contains expected result
        if expected_roman in answer:
            correct += 1
        else:
            errors.append({
                'problem': f"{a} + {b}",
                'prompt': prompt,
                'expected': expected_roman,
                'generated': answer,
                'full_output': generated
            })

        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, errors

def evaluate_subtraction_problems(model, tokenizer, device, num_problems=100,
                                  min_val=1, max_val=50, use_additive=False):
    """Evaluate model on subtraction problems"""
    correct = 0
    total = 0
    errors = []

    while total < num_problems:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

        # Ensure a > b for valid subtraction (must be strictly greater)
        if a <= b:
            a, b = b, a

        # Skip if still equal (can happen with small ranges)
        if a == b:
            continue

        total += 1

        if use_additive:
            a_roman = int_to_roman_additive(a)
            b_roman = int_to_roman_additive(b)
            expected_roman = int_to_roman_additive(a - b)
        else:
            a_roman = int_to_roman_subtractive(a)
            b_roman = int_to_roman_subtractive(b)
            expected_roman = int_to_roman_subtractive(a - b)

        prompt = f"{a_roman} - {b_roman} |"
        generated = generate_text(model, prompt, tokenizer, device)

        # Extract final answer
        answer = extract_final_answer(generated)

        # Check if answer contains expected result
        if expected_roman in answer:
            correct += 1
        else:
            errors.append({
                'problem': f"{a} - {b}",
                'prompt': prompt,
                'expected': expected_roman,
                'generated': answer,
                'full_output': generated
            })

    accuracy = correct / total if total > 0 else 0
    return accuracy, errors

def evaluate_range_generalization(model, tokenizer, device, ranges, use_additive=False):
    """Evaluate model on different number ranges (addition only)"""
    results = {}

    for range_name, (min_val, max_val) in ranges.items():
        accuracy, errors = evaluate_addition_problems(
            model, tokenizer, device,
            num_problems=50,
            min_val=min_val,
            max_val=max_val,
            use_additive=use_additive
        )
        results[range_name] = {
            'accuracy': accuracy,
            'error_count': len(errors),
            'sample_errors': errors[:3]  # Keep first 3 errors as examples
        }

    return results

def evaluate_algorithmic_steps(model, tokenizer, device, num_problems=20):
    """Evaluate if model shows correct algorithmic steps"""
    step_accuracy = {
        'convert_additive': 0,
        'concatenate': 0,
        'sort': 0,
        'reduce': 0,
        'final_answer': 0
    }
    total = num_problems

    for _ in range(num_problems):
        a = random.randint(10, 30)
        b = random.randint(10, 30)

        a_roman = int_to_roman_subtractive(a)
        b_roman = int_to_roman_subtractive(b)

        prompt = f"{a_roman} + {b_roman} |"
        generated = generate_text(model, prompt, tokenizer, device, max_new_tokens=300)

        # Check algorithmic steps
        a_add = subtractive_to_additive(a_roman)
        b_add = subtractive_to_additive(b_roman)

        if a_add in generated and b_add in generated:
            step_accuracy['convert_additive'] += 1

        concat = a_add + b_add
        if concat in generated:
            step_accuracy['concatenate'] += 1

        sorted_roman = sort_roman(concat)
        if sorted_roman in generated:
            step_accuracy['sort'] += 1

        reduced = reduce_roman(sorted_roman)
        if reduced in generated:
            step_accuracy['reduce'] += 1

        final = int_to_roman_subtractive(a + b)
        if final in generated:
            step_accuracy['final_answer'] += 1

    # Convert to percentages
    for key in step_accuracy:
        step_accuracy[key] = step_accuracy[key] / total

    return step_accuracy

# ============================================================================
# METADATA AND CHECKPOINT LOADING
# ============================================================================

def load_checkpoint_with_metadata(checkpoint_path, device):
    """Load model checkpoint and its metadata file"""
    checkpoint_path = Path(checkpoint_path)

    # Load metadata
    metadata_path = checkpoint_path.with_suffix('.json')
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"This checkpoint was likely created with an older version of the trainer.\n"
            f"Please retrain the model with the updated trainer to generate metadata files."
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load model state dict
    state_dict = torch.load(checkpoint_path, map_location=device)

    return metadata, state_dict

def get_tokenizer_from_metadata(metadata):
    """Get the appropriate tokenizer based on metadata"""
    tokenizer_type = metadata.get('tokenizer_type', 'CharTokenizer')

    if tokenizer_type == 'CharTokenizer':
        return CharTokenizer()
    elif tokenizer_type == 'RomanTokenizer':
        return RomanTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

# ============================================================================
# EXHAUSTIVE TESTING
# ============================================================================

def exhaustive_test(model, tokenizer, device, max_num, use_additive=False):
    """Test all combinations of numbers up to max_num"""
    print("\n" + "="*60)
    print(f"EXHAUSTIVE TEST (1 to {max_num})")
    print("="*60)

    total_addition = 0
    correct_addition = 0
    failed_addition = []

    total_subtraction = 0
    correct_subtraction = 0
    failed_subtraction = []

    print("\nTesting all addition combinations...")
    # Test all addition combinations
    for a in range(1, max_num + 1):
        for b in range(1, max_num + 1):
            try:
                if use_additive:
                    a_roman = int_to_roman_additive(a)
                    b_roman = int_to_roman_additive(b)
                    expected = int_to_roman_additive(a + b)
                else:
                    a_roman = int_to_roman_subtractive(a)
                    b_roman = int_to_roman_subtractive(b)
                    expected = int_to_roman_subtractive(a + b)

                prompt = f"{a_roman} + {b_roman} |"
                generated = generate_text(model, prompt, tokenizer, device)

                total_addition += 1
                if expected in generated:
                    correct_addition += 1
                else:
                    failed_addition.append({
                        'a': a,
                        'b': b,
                        'problem': f"{a} + {b} = {a + b}",
                        'prompt': prompt,
                        'expected': expected,
                        'generated': extract_final_answer(generated)
                    })
            except ValueError:
                # Result out of range
                continue

        # Progress indicator
        if a % 5 == 0:
            print(f"  Progress: {a}/{max_num} ({correct_addition}/{total_addition} correct so far)")

    print(f"\nTesting all subtraction combinations...")
    # Test all subtraction combinations (where a > b)
    for a in range(2, max_num + 1):  # Start at 2 since we need a > b
        for b in range(1, a):  # b must be less than a
            try:
                if use_additive:
                    a_roman = int_to_roman_additive(a)
                    b_roman = int_to_roman_additive(b)
                    expected = int_to_roman_additive(a - b)
                else:
                    a_roman = int_to_roman_subtractive(a)
                    b_roman = int_to_roman_subtractive(b)
                    expected = int_to_roman_subtractive(a - b)

                prompt = f"{a_roman} - {b_roman} |"
                generated = generate_text(model, prompt, tokenizer, device)

                total_subtraction += 1
                if expected in generated:
                    correct_subtraction += 1
                else:
                    failed_subtraction.append({
                        'a': a,
                        'b': b,
                        'problem': f"{a} - {b} = {a - b}",
                        'prompt': prompt,
                        'expected': expected,
                        'generated': extract_final_answer(generated)
                    })
            except ValueError:
                # Result out of range
                continue

        # Progress indicator
        if a % 5 == 0:
            print(f"  Progress: {a}/{max_num} ({correct_subtraction}/{total_subtraction} correct so far)")

    # Display results
    print("\n" + "="*60)
    print("EXHAUSTIVE TEST RESULTS")
    print("="*60)

    add_accuracy = correct_addition / total_addition if total_addition > 0 else 0
    sub_accuracy = correct_subtraction / total_subtraction if total_subtraction > 0 else 0

    print(f"\nAddition:")
    print(f"  Total problems: {total_addition}")
    print(f"  Correct: {correct_addition}")
    print(f"  Failed: {len(failed_addition)}")
    print(f"  Accuracy: {add_accuracy:.2%}")

    print(f"\nSubtraction:")
    print(f"  Total problems: {total_subtraction}")
    print(f"  Correct: {correct_subtraction}")
    print(f"  Failed: {len(failed_subtraction)}")
    print(f"  Accuracy: {sub_accuracy:.2%}")

    print(f"\nOverall:")
    overall_total = total_addition + total_subtraction
    overall_correct = correct_addition + correct_subtraction
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    print(f"  Total problems: {overall_total}")
    print(f"  Correct: {overall_correct}")
    print(f"  Failed: {len(failed_addition) + len(failed_subtraction)}")
    print(f"  Accuracy: {overall_accuracy:.2%}")

    # Show sample failures
    if failed_addition:
        print(f"\nSample addition failures ({min(5, len(failed_addition))} of {len(failed_addition)}):")
        for i, failure in enumerate(failed_addition[:5]):
            print(f"\n  {i+1}. {failure['problem']}")
            print(f"     Prompt: {failure['prompt']}")
            print(f"     Expected: {failure['expected']}")
            print(f"     Generated: {failure['generated']}")

    if failed_subtraction:
        print(f"\nSample subtraction failures ({min(5, len(failed_subtraction))} of {len(failed_subtraction)}):")
        for i, failure in enumerate(failed_subtraction[:5]):
            print(f"\n  {i+1}. {failure['problem']}")
            print(f"     Prompt: {failure['prompt']}")
            print(f"     Expected: {failure['expected']}")
            print(f"     Generated: {failure['generated']}")

    # Save detailed results
    results = {
        'exhaustive_test': {
            'max_num': max_num,
            'addition': {
                'total': total_addition,
                'correct': correct_addition,
                'accuracy': add_accuracy,
                'failures': failed_addition
            },
            'subtraction': {
                'total': total_subtraction,
                'correct': correct_subtraction,
                'accuracy': sub_accuracy,
                'failures': failed_subtraction
            },
            'overall': {
                'total': overall_total,
                'correct': overall_correct,
                'accuracy': overall_accuracy
            }
        }
    }

    return results

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(model, tokenizer, device):
    """Interactive prompting mode"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nEnter Roman numeral arithmetic prompts.")
    print("Examples:")
    print("  X + V |")
    print("  XX - III |")
    print("  C + L |")
    print("\nType 'quit' or 'exit' to end.\n")

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break

            if not prompt:
                continue

            # Add pipe separator if not present
            if not prompt.endswith('|'):
                prompt = prompt + ' |'

            # Generate response
            print("\nGenerating...", end='', flush=True)
            generated = generate_text(model, prompt, tokenizer, device, max_new_tokens=300)
            print("\r" + " " * 20 + "\r", end='')  # Clear the "Generating..." message

            print(f"Generated: {generated}")
            print()

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print()

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Roman Numeral Transformer Checkpoints')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file (.pt)')
    parser.add_argument('--additive', action='store_true', help='Use additive notation (VIIII vs IX)')
    parser.add_argument('--num-problems', type=int, default=100, help='Number of test problems')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file for results')
    parser.add_argument('--interactive', action='store_true', help='Interactive prompting mode')
    parser.add_argument('--exhaustive', type=int, metavar='MAX', help='Exhaustive test: test all combinations up to MAX')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print()

    # Load model configuration from metadata
    print("Loading model metadata...")
    try:
        metadata, state_dict = load_checkpoint_with_metadata(args.checkpoint, device)
        config = metadata['model_config']
        print("Metadata loaded successfully:")
        print(f"  - Model type: {metadata.get('model_type', 'Unknown')}")
        print(f"  - Tokenizer type: {metadata.get('tokenizer_type', 'Unknown')}")
        print(f"  - Training format: {metadata.get('training_config', {}).get('format', 'Unknown')}")
        print(f"\n  Model Architecture:")
        print(f"  - vocab_size: {config['vocab_size']}")
        print(f"  - d_model: {config['d_model']}")
        print(f"  - nhead: {config['nhead']}")
        print(f"  - num_layers: {config['num_layers']}")
        print(f"  - dim_feedforward: {config['dim_feedforward']}")
        print(f"  - max_seq_length: {config['max_seq_length']}")
    except Exception as e:
        print(f"ERROR: Failed to load metadata: {e}")
        return

    # Get tokenizer from metadata
    try:
        tokenizer = get_tokenizer_from_metadata(metadata)
        print(f"\nTokenizer initialized: {metadata.get('tokenizer_type')}")
    except Exception as e:
        print(f"ERROR: Failed to initialize tokenizer: {e}")
        return

    # Get training format for display
    training_format = metadata.get('training_config', {}).get('format', 'unknown')
    print(f"\nTraining format: {training_format}")
    print(f"Notation: {'Additive (VIIII)' if args.additive else 'Subtractive (IX)'}")
    print()

    # Initialize model with config from metadata
    model = SimpleTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        max_seq_length=config['max_seq_length']
    ).to(device)

    # Load checkpoint
    try:
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return

    model.eval()

    # Enter interactive mode if requested
    if args.interactive:
        interactive_mode(model, tokenizer, device)
        return

    # Run exhaustive test if requested
    if args.exhaustive:
        results = exhaustive_test(model, tokenizer, device, args.exhaustive, args.additive)
        # Save results
        output_file = args.output or f'exhaustive_test_{args.exhaustive}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
        return

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    results = {}

    # 1. Basic accuracy (both operations)
    print("\n1. Basic Accuracy Test")
    print("-" * 40)

    # Test addition
    add_accuracy, add_errors = evaluate_addition_problems(
        model, tokenizer, device,
        num_problems=args.num_problems // 2,
        use_additive=args.additive
    )
    print(f"Addition: {add_accuracy:.2%} ({int(add_accuracy * (args.num_problems // 2))}/{args.num_problems // 2})")

    # Test subtraction
    sub_accuracy, sub_errors = evaluate_subtraction_problems(
        model, tokenizer, device,
        num_problems=args.num_problems // 2,
        use_additive=args.additive
    )
    print(f"Subtraction: {sub_accuracy:.2%} ({int(sub_accuracy * (args.num_problems // 2))}/{args.num_problems // 2})")

    # Overall accuracy
    overall_accuracy = (add_accuracy + sub_accuracy) / 2
    print(f"Overall: {overall_accuracy:.2%}")

    results['addition_accuracy'] = add_accuracy
    results['subtraction_accuracy'] = sub_accuracy
    results['overall_accuracy'] = overall_accuracy

    # Show sample errors from both
    if add_errors:
        print(f"\nSample addition errors ({min(2, len(add_errors))} of {len(add_errors)}):")
        for i, error in enumerate(add_errors[:2]):
            print(f"\n  Error {i+1}:")
            print(f"    Problem: {error['problem']}")
            print(f"    Prompt: {error['prompt']}")
            print(f"    Expected: {error['expected']}")
            print(f"    Generated: {error['generated']}")

    if sub_errors:
        print(f"\nSample subtraction errors ({min(2, len(sub_errors))} of {len(sub_errors)}):")
        for i, error in enumerate(sub_errors[:2]):
            print(f"\n  Error {i+1}:")
            print(f"    Problem: {error['problem']}")
            print(f"    Prompt: {error['prompt']}")
            print(f"    Expected: {error['expected']}")
            print(f"    Generated: {error['generated']}")

    # 2. Range generalization
    print("\n2. Range Generalization")
    print("-" * 40)
    ranges = {
        'Small (1-10)': (1, 10),
        'Medium (10-50)': (10, 50),
        'Large (50-100)': (50, 100),
        'Very Large (100-500)': (100, 500)
    }

    range_results = evaluate_range_generalization(model, tokenizer, device, ranges, args.additive)
    results['range_generalization'] = {}

    for range_name, result in range_results.items():
        print(f"{range_name:20s}: {result['accuracy']:.2%}")
        results['range_generalization'][range_name] = result['accuracy']

    # 3. Algorithmic steps (only for models trained with intermediate/full format)
    if training_format in ['intermediate', 'full']:
        print("\n3. Algorithmic Step Accuracy")
        print("-" * 40)
        step_acc = evaluate_algorithmic_steps(model, tokenizer, device)
        results['algorithmic_steps'] = step_acc

        print(f"Convert to additive: {step_acc['convert_additive']:.2%}")
        print(f"Concatenate:         {step_acc['concatenate']:.2%}")
        print(f"Sort:                {step_acc['sort']:.2%}")
        print(f"Reduce:              {step_acc['reduce']:.2%}")
        print(f"Final answer:        {step_acc['final_answer']:.2%}")

    # 4. Sample generations (mixed operations)
    print("\n4. Sample Generations")
    print("-" * 40)
    samples = []

    # Generate 3 addition samples
    for _ in range(3):
        a = random.randint(10, 40)
        b = random.randint(10, 40)

        if args.additive:
            a_roman = int_to_roman_additive(a)
            b_roman = int_to_roman_additive(b)
            expected = int_to_roman_additive(a + b)
        else:
            a_roman = int_to_roman_subtractive(a)
            b_roman = int_to_roman_subtractive(b)
            expected = int_to_roman_subtractive(a + b)

        prompt = f"{a_roman} + {b_roman} |"
        generated = generate_text(model, prompt, tokenizer, device)

        print(f"\nAddition: {a} + {b}")
        print(f"Prompt: {prompt}")
        print(f"Expected: {expected}")
        print(f"Generated: {generated[:200]}...")  # Truncate if too long

        samples.append({
            'operation': 'addition',
            'problem': f"{a} + {b}",
            'prompt': prompt,
            'expected': expected,
            'generated': generated
        })

    # Generate 2 subtraction samples
    for _ in range(2):
        a = random.randint(30, 80)
        b = random.randint(10, 29)

        if args.additive:
            a_roman = int_to_roman_additive(a)
            b_roman = int_to_roman_additive(b)
            expected = int_to_roman_additive(a - b)
        else:
            a_roman = int_to_roman_subtractive(a)
            b_roman = int_to_roman_subtractive(b)
            expected = int_to_roman_subtractive(a - b)

        prompt = f"{a_roman} - {b_roman} |"
        generated = generate_text(model, prompt, tokenizer, device)

        print(f"\nSubtraction: {a} - {b}")
        print(f"Prompt: {prompt}")
        print(f"Expected: {expected}")
        print(f"Generated: {generated[:200]}...")  # Truncate if too long

        samples.append({
            'operation': 'subtraction',
            'problem': f"{a} - {b}",
            'prompt': prompt,
            'expected': expected,
            'generated': generated
        })

    results['samples'] = samples

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"Results saved to: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()
