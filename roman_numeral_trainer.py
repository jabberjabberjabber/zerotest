import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math
from pathlib import Path
import argparse
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
# GENERATE TRAINING EXAMPLES - ADDITION
# ============================================================================

def generate_addition_ultra_simple(a, b):
    """Simplest format: A + B = C"""
    a_roman = int_to_roman_subtractive(a)
    b_roman = int_to_roman_subtractive(b)
    c_roman = int_to_roman_subtractive(a + b)

    return f"{a_roman} + {b_roman} = {c_roman}"

def generate_addition_with_intermediate(a, b):
    """Show sorted concatenation: A + B | sorted | = C"""
    a_roman = int_to_roman_subtractive(a)
    b_roman = int_to_roman_subtractive(b)

    a_add = subtractive_to_additive(a_roman)
    b_add = subtractive_to_additive(b_roman)
    concat = a_add + b_add
    sorted_roman = sort_roman(concat)

    c_roman = int_to_roman_subtractive(a + b)

    return f"{a_roman} + {b_roman} | {sorted_roman} | = {c_roman}"

def generate_addition_full_steps(a, b):
    """Show all intermediate steps inline"""
    a_roman = int_to_roman_subtractive(a)
    b_roman = int_to_roman_subtractive(b)

    a_add = subtractive_to_additive(a_roman)
    b_add = subtractive_to_additive(b_roman)
    concat = a_add + b_add
    sorted_roman = sort_roman(concat)
    reduced = reduce_roman(sorted_roman)
    final = additive_to_subtractive(reduced)

    return f"{a_roman} + {b_roman} | {a_add} {b_add} | {sorted_roman} | {reduced} | {final}"

# ============================================================================
# GENERATE TRAINING EXAMPLES - SUBTRACTION
# ============================================================================

def generate_subtraction_ultra_simple(a, b):
    """Simplest format: A - B = C"""
    if a <= b:
        # Skip invalid subtractions (would give zero or negative)
        raise ValueError("Invalid subtraction: a must be > b")

    a_roman = int_to_roman_subtractive(a)
    b_roman = int_to_roman_subtractive(b)
    c_roman = int_to_roman_subtractive(a - b)

    return f"{a_roman} - {b_roman} = {c_roman}"

def generate_subtraction_with_intermediate(a, b):
    """Show intermediate steps: A - B | after_eliminations | = C"""
    if a <= b:
        raise ValueError("Invalid subtraction: a must be > b")

    a_roman = int_to_roman_subtractive(a)
    b_roman = int_to_roman_subtractive(b)

    # Show the state after eliminating common symbols
    a_add = subtractive_to_additive(a_roman)
    b_add = subtractive_to_additive(b_roman)

    # Simulate elimination for display
    a_temp = list(a_add)
    b_temp = list(b_add)
    for char in b_add:
        if char in a_temp:
            a_temp.remove(char)
    intermediate = ''.join(a_temp)

    c_roman = int_to_roman_subtractive(a - b)

    return f"{a_roman} - {b_roman} | {intermediate} | = {c_roman}"

def generate_subtraction_full_steps(a, b):
    """Show all intermediate steps inline"""
    if a <= b:
        raise ValueError("Invalid subtraction: a must be > b")

    a_roman = int_to_roman_subtractive(a)
    b_roman = int_to_roman_subtractive(b)

    a_add = subtractive_to_additive(a_roman)
    b_add = subtractive_to_additive(b_roman)

    # Get intermediate state (after eliminations)
    a_temp = list(a_add)
    b_temp = list(b_add)
    for char in b_add:
        if char in a_temp:
            a_temp.remove(char)
    intermediate = ''.join(a_temp)

    result = subtract_roman(a_roman, b_roman)

    return f"{a_roman} - {b_roman} | {a_add} {b_add} | {intermediate} | {result}"

# ============================================================================
# TOKENIZER
# ============================================================================

class CharTokenizer:
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

# ============================================================================
# DATASET
# ============================================================================

class RomanDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx] + '<EOS>'
        tokens = self.tokenizer.encode(text)

        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_idx] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens[:-1], tokens[1:]

# ============================================================================
# MODEL
# ============================================================================

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, max_seq_length=128, dropout=0.1):
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
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, tokenizer):
    model.train()
    total_loss = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tokenizer.pad_idx
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def generate_text(model, prompt, tokenizer, device, max_new_tokens=100):
    """Generate text from a prompt"""
    model.eval()

    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
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

def evaluate_accuracy(model, tokenizer, device, num_tests=50, min_val=1, max_val=50):
    """Evaluate model accuracy on addition"""
    correct = 0

    for _ in range(num_tests):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

        a_roman = int_to_roman_subtractive(a)
        b_roman = int_to_roman_subtractive(b)
        expected = int_to_roman_subtractive(a + b)

        prompt = f"{a_roman} + {b_roman} |"
        generated = generate_text(model, prompt, tokenizer, device)

        if expected in generated:
            correct += 1

    return correct / num_tests

def evaluate_subtraction_accuracy(model, tokenizer, device, num_tests=50, min_val=1, max_val=50):
    """Evaluate model accuracy on subtraction"""
    correct = 0
    tested = 0

    while tested < num_tests:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

        # Ensure a > b for valid subtraction (must be strictly greater)
        if a <= b:
            a, b = b, a

        # Skip if still equal (can happen with small ranges)
        if a == b:
            continue

        tested += 1

        a_roman = int_to_roman_subtractive(a)
        b_roman = int_to_roman_subtractive(b)
        expected = int_to_roman_subtractive(a - b)

        prompt = f"{a_roman} - {b_roman} |"
        generated = generate_text(model, prompt, tokenizer, device)

        if expected in generated:
            correct += 1

    return correct / num_tests

# ============================================================================
# CURRICULUM LEARNING
# ============================================================================

def generate_curriculum_data(stage, examples_per_stage, add_fn, sub_fn):
    """Generate data with increasing difficulty (mixed operations)"""
    curricula = {
        1: (1, 5),      # Very small numbers
        2: (1, 10),     # Small numbers
        3: (5, 20),     # Small to medium
        4: (10, 30),    # Medium
        5: (10, 50),    # Medium to large
        6: (20, 100),   # Large
        7: (50, 200),   # Very large
    }

    min_val, max_val = curricula.get(stage, (1, 50))
    examples = []

    for _ in range(examples_per_stage):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

        try:
            # Randomly choose addition or subtraction
            if random.random() < 0.5:
                example = add_fn(a, b)
            else:
                # For subtraction, ensure a > b
                if a > b:
                    example = sub_fn(a, b)
                else:
                    example = sub_fn(b, a)
            examples.append(example)
        except ValueError:
            # Skip if result out of range
            continue

    return examples

# ============================================================================
# METADATA MANAGEMENT
# ============================================================================

def save_checkpoint_with_metadata(model, checkpoint_path, metadata):
    """Save model checkpoint and its metadata file"""
    # Save model weights
    torch.save(model.state_dict(), checkpoint_path)

    # Save metadata as JSON
    metadata_path = checkpoint_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Improved Roman Numeral Algorithm Trainer')
    parser.add_argument('--format', type=str, choices=['simple', 'intermediate', 'full'],
                       default='intermediate', help='Training format complexity')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--dim-feedforward', type=int, default=2048, help='FFN dimension')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--epochs', type=int, default=300, help='Total epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--examples', type=int, default=10000, help='Training examples')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Format: {args.format}")
    print(f"Curriculum learning: {args.curriculum}")
    print(f"Model size: d_model={args.d_model}, layers={args.num_layers}, heads={args.nhead}")
    print()

    # Choose format functions for addition and subtraction
    addition_functions = {
        'simple': generate_addition_ultra_simple,
        'intermediate': generate_addition_with_intermediate,
        'full': generate_addition_full_steps
    }
    subtraction_functions = {
        'simple': generate_subtraction_ultra_simple,
        'intermediate': generate_subtraction_with_intermediate,
        'full': generate_subtraction_full_steps
    }
    add_fn = addition_functions[args.format]
    sub_fn = subtraction_functions[args.format]

    # Initialize tokenizer
    tokenizer = CharTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Initialize model
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_seq_length=128
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Create metadata for this training run
    metadata = {
        'model_config': {
            'vocab_size': tokenizer.vocab_size,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'max_seq_length': 128,
            'dropout': 0.1
        },
        'training_config': {
            'format': args.format,
            'curriculum': args.curriculum,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'num_examples': args.examples
        },
        'tokenizer_type': 'CharTokenizer',
        'model_type': 'SimpleTransformer'
    }

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.curriculum:
        print("="*60)
        print("CURRICULUM TRAINING")
        print("="*60)

        # Train through curriculum stages
        for stage in range(1, 8):
            print(f"\n--- Stage {stage}/7 ---")

            # Generate curriculum data
            examples = generate_curriculum_data(stage, args.examples // 7, add_fn, sub_fn)
            print(f"Generated {len(examples)} examples")
            if examples:
                add_examples = [ex for ex in examples if ' + ' in ex]
                sub_examples = [ex for ex in examples if ' - ' in ex]
                if add_examples:
                    print(f"Addition example: {add_examples[0]}")
                if sub_examples:
                    print(f"Subtraction example: {sub_examples[0]}")

            # Create dataset
            dataset = RomanDataset(examples, tokenizer)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            # Train for fewer epochs per stage
            epochs_per_stage = args.epochs // 7
            for epoch in range(epochs_per_stage):
                loss = train_epoch(model, dataloader, optimizer, device, tokenizer)

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs_per_stage}: Loss = {loss:.4f}")

                    # Quick test on both operations
                    add_acc = evaluate_accuracy(model, tokenizer, device, num_tests=10)
                    sub_acc = evaluate_subtraction_accuracy(model, tokenizer, device, num_tests=10)
                    print(f"  Addition: {add_acc:.2%}, Subtraction: {sub_acc:.2%}")

            # Save checkpoint after each stage
            checkpoint_path = checkpoint_dir / f"curriculum_stage_{stage}.pt"
            stage_metadata = metadata.copy()
            stage_metadata['curriculum_stage'] = stage
            save_checkpoint_with_metadata(model, checkpoint_path, stage_metadata)
            print(f"Saved checkpoint: {checkpoint_path}")

    else:
        print("="*60)
        print("STANDARD TRAINING")
        print("="*60)

        # Generate all training data (50% addition, 50% subtraction)
        print(f"\nGenerating {args.examples} training examples (mixed operations)...")
        train_examples = []

        for _ in range(args.examples):
            a = random.randint(1, 1500)
            b = random.randint(1, 1500)

            try:
                # Randomly choose addition or subtraction
                if random.random() < 0.5:
                    example = add_fn(a, b)
                else:
                    # For subtraction, ensure a > b
                    if a > b:
                        example = sub_fn(a, b)
                    else:
                        example = sub_fn(b, a)
                train_examples.append(example)
            except ValueError:
                continue

        print(f"Generated {len(train_examples)} examples")
        print(f"Addition example: {[ex for ex in train_examples if ' + ' in ex][0]}")
        print(f"Subtraction example: {[ex for ex in train_examples if ' - ' in ex][0]}")
        print()

        # Create dataset
        dataset = RomanDataset(train_examples, tokenizer)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Training loop
        best_acc = 0
        print("Training...")
        for epoch in range(args.epochs):
            loss = train_epoch(model, dataloader, optimizer, device, tokenizer)

            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{args.epochs}: Loss = {loss:.4f}")

                # Evaluate both operations
                add_acc = evaluate_accuracy(model, tokenizer, device, num_tests=50)
                sub_acc = evaluate_subtraction_accuracy(model, tokenizer, device, num_tests=50)
                acc = (add_acc + sub_acc) / 2  # Average accuracy
                print(f"Addition accuracy: {add_acc:.2%}")
                print(f"Subtraction accuracy: {sub_acc:.2%}")
                print(f"Overall accuracy: {acc:.2%}")

                # Show addition sample
                test_a = random.randint(1, 100)
                test_b = random.randint(1, 100)
                test_prompt = f"{int_to_roman_subtractive(test_a)} + {int_to_roman_subtractive(test_b)} |"
                generated = generate_text(model, test_prompt, tokenizer, device)
                expected = int_to_roman_subtractive(test_a + test_b)
                print(f"Addition sample: {test_a} + {test_b} = {test_a + test_b}")
                print(f"  Prompt: {test_prompt}")
                print(f"  Generated: {generated}")
                print(f"  Expected: {expected}")

                # Show subtraction sample
                test_a = random.randint(50, 100)
                test_b = random.randint(1, 49)
                test_prompt = f"{int_to_roman_subtractive(test_a)} - {int_to_roman_subtractive(test_b)} |"
                generated = generate_text(model, test_prompt, tokenizer, device)
                expected = int_to_roman_subtractive(test_a - test_b)
                print(f"Subtraction sample: {test_a} - {test_b} = {test_a - test_b}")
                print(f"  Prompt: {test_prompt}")
                print(f"  Generated: {generated}")
                print(f"  Expected: {expected}")

                # Save best model
                if acc > best_acc:
                    best_acc = acc
                    checkpoint_path = checkpoint_dir / "best_model.pt"
                    best_metadata = metadata.copy()
                    best_metadata['epoch'] = epoch + 1
                    best_metadata['accuracy'] = acc
                    save_checkpoint_with_metadata(model, checkpoint_path, best_metadata)
                    print(f"New best! Saved to {checkpoint_path}")

            # Save periodic checkpoints
            if (epoch + 1) % 50 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                epoch_metadata = metadata.copy()
                epoch_metadata['epoch'] = epoch + 1
                save_checkpoint_with_metadata(model, checkpoint_path, epoch_metadata)
                print(f"Checkpoint saved: {checkpoint_path}")

    # Final save
    final_path = checkpoint_dir / "final_model.pt"
    final_metadata = metadata.copy()
    final_metadata['training_complete'] = True
    save_checkpoint_with_metadata(model, final_path, final_metadata)
    print(f"\nFinal model saved: {final_path}")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    for range_name, (min_val, max_val) in [('Small (1-10)', (1, 10)),
                                             ('Medium (10-50)', (10, 50)),
                                             ('Large (50-100)', (50, 100))]:
        add_acc = evaluate_accuracy(model, tokenizer, device, num_tests=50,
                                    min_val=min_val, max_val=max_val)
        sub_acc = evaluate_subtraction_accuracy(model, tokenizer, device, num_tests=50,
                                                 min_val=min_val, max_val=max_val)
        print(f"{range_name}:")
        print(f"  Addition: {add_acc:.2%}")
        print(f"  Subtraction: {sub_acc:.2%}")
        print(f"  Overall: {(add_acc + sub_acc) / 2:.2%}")

if __name__ == "__main__":
    main()
