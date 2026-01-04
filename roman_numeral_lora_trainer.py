"""
Roman Numeral LoRA Trainer - Zero Concept Addition

This script implements LoRA (Low-Rank Adaptation) training to add the zero concept
('N') to a pre-trained Roman numeral arithmetic model.

Architecture:
  1. Load a pre-trained base model (frozen weights)
  2. Add small trainable LoRA matrices to specific layers
  3. Train only the LoRA parameters on zero-concept dataset
  4. Save only the LoRA weights (not the full model)

Zero Concept Operations:
  - N + N = N  (zero plus zero)
  - N + X = X  (zero plus number)
  - X + N = X  (number plus zero)
  - X - N = X  (number minus zero)
  - X - X = N  (result is zero)
  - N - N = N  (zero minus zero)

Usage:
  python roman_numeral_lora_trainer.py \\
    --base-model checkpoints/best_model.pt \\
    --format intermediate \\
    --lora-rank 8 \\
    --lora-alpha 16 \\
    --epochs 100 \\
    --examples 5000

Output:
  - lora_checkpoints/best_lora.pt (LoRA weights only)
  - lora_checkpoints/best_lora.json (metadata)

See LORA_DOCUMENTATION.md for detailed information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math
from pathlib import Path
import argparse
import json
from roman_numeral_trainer import (
    CharTokenizer,
    RomanDataset,
    SimpleTransformer,
    int_to_roman_subtractive,
    generate_text
)

# ============================================================================
# LORA IMPLEMENTATION
# ============================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for linear transformations"""
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """Apply LoRA transformation: x @ (A @ B) * scaling"""
        return self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    """Wraps a frozen linear layer with LoRA adaptation"""
    def __init__(self, linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.0, target_modules=None):
    """Apply LoRA to specified modules in the transformer model

    Args:
        model: The base SimpleTransformer model
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout
        target_modules: Which modules to adapt. Options:
            - None or 'output': Only output layer (default, recommended)
            - 'attn_output': Attention output projections + model output
            - 'all': All possible layers (experimental)

    Note: PyTorch's nn.MultiheadAttention uses combined in_proj_weight for Q/K/V,
    so we can only apply LoRA to the attention output projection (out_proj).
    """
    if target_modules is None:
        target_modules = 'output'

    # Apply LoRA to attention output projections
    if target_modules in ['attn_output', 'all']:
        for layer_idx, layer in enumerate(model.transformer.layers):
            self_attn = layer.self_attn

            # Apply to attention output projection
            if hasattr(self_attn, 'out_proj') and isinstance(self_attn.out_proj, nn.Linear):
                self_attn.out_proj = LinearWithLoRA(
                    self_attn.out_proj, rank, alpha, dropout
                )
                print(f"  Applied LoRA to layer {layer_idx} attention output")

    # Apply LoRA to model output layer (recommended for learning new concepts)
    if target_modules in ['output', 'attn_output', 'all']:
        model.output = LinearWithLoRA(model.output, rank, alpha, dropout)
        print(f"  Applied LoRA to model output layer")

    return model


def get_lora_parameters(model):
    """Extract only LoRA parameters from model"""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def count_parameters(model, trainable_only=True):
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


# ============================================================================
# ZERO (N) DATASET GENERATION
# ============================================================================

def generate_zero_addition(a, b, format='intermediate'):
    """Generate addition examples involving zero (N)

    Cases:
    - N + N = N
    - N + X = X
    - X + N = X
    """
    # Determine which operand(s) are zero
    a_is_zero = (a == 0)
    b_is_zero = (b == 0)

    a_roman = 'N' if a_is_zero else int_to_roman_subtractive(a)
    b_roman = 'N' if b_is_zero else int_to_roman_subtractive(b)
    result = a + b
    result_roman = 'N' if result == 0 else int_to_roman_subtractive(result)

    if format == 'simple':
        return f"{a_roman} + {b_roman} = {result_roman}"
    elif format == 'intermediate':
        # For zero operations, intermediate step is simple
        if a_is_zero and b_is_zero:
            return f"{a_roman} + {b_roman} | N | = {result_roman}"
        elif a_is_zero:
            return f"{a_roman} + {b_roman} | {b_roman} | = {result_roman}"
        elif b_is_zero:
            return f"{a_roman} + {b_roman} | {a_roman} | = {result_roman}"
    elif format == 'full':
        # Similar to intermediate for zero cases
        return generate_zero_addition(a, b, format='intermediate')

    return f"{a_roman} + {b_roman} = {result_roman}"


def generate_zero_subtraction(a, b, format='intermediate'):
    """Generate subtraction examples involving zero (N)

    Cases:
    - X - N = X (subtracting zero)
    - X - X = N (result is zero)
    - N - N = N (zero minus zero)
    """
    # Determine which operand(s) are zero
    a_is_zero = (a == 0)
    b_is_zero = (b == 0)

    a_roman = 'N' if a_is_zero else int_to_roman_subtractive(a)
    b_roman = 'N' if b_is_zero else int_to_roman_subtractive(b)
    result = a - b
    result_roman = 'N' if result == 0 else int_to_roman_subtractive(result)

    if format == 'simple':
        return f"{a_roman} - {b_roman} = {result_roman}"
    elif format == 'intermediate':
        # For zero operations, intermediate step is simple
        if a_is_zero and b_is_zero:
            return f"{a_roman} - {b_roman} | N | = {result_roman}"
        elif b_is_zero:
            return f"{a_roman} - {b_roman} | {a_roman} | = {result_roman}"
        elif a == b:  # X - X = N
            return f"{a_roman} - {b_roman} | N | = {result_roman}"
    elif format == 'full':
        return generate_zero_subtraction(a, b, format='intermediate')

    return f"{a_roman} - {b_roman} = {result_roman}"


def generate_zero_dataset(num_examples, format='intermediate'):
    """Generate dataset with zero (N) operations

    Distribution:
    - N + N = N
    - N + X = X
    - X + N = X
    - X - N = X
    - X - X = N
    - N - N = N
    """
    examples = []
    examples_per_type = num_examples // 6

    # 1. N + N = N
    for _ in range(examples_per_type):
        examples.append(generate_zero_addition(0, 0, format))

    # 2. N + X = X
    for _ in range(examples_per_type):
        x = random.randint(1, 3999)
        examples.append(generate_zero_addition(0, x, format))

    # 3. X + N = X
    for _ in range(examples_per_type):
        x = random.randint(1, 3999)
        examples.append(generate_zero_addition(x, 0, format))

    # 4. X - N = X
    for _ in range(examples_per_type):
        x = random.randint(1, 3999)
        examples.append(generate_zero_subtraction(x, 0, format))

    # 5. X - X = N
    for _ in range(examples_per_type):
        x = random.randint(1, 3999)
        examples.append(generate_zero_subtraction(x, x, format))

    # 6. N - N = N
    for _ in range(examples_per_type):
        examples.append(generate_zero_subtraction(0, 0, format))

    # Fill remaining to reach exact num_examples
    while len(examples) < num_examples:
        operation_type = random.choice(['add', 'sub'])
        if operation_type == 'add':
            choice = random.randint(0, 2)
            if choice == 0:
                examples.append(generate_zero_addition(0, 0, format))
            elif choice == 1:
                x = random.randint(1, 3999)
                examples.append(generate_zero_addition(0, x, format))
            else:
                x = random.randint(1, 3999)
                examples.append(generate_zero_addition(x, 0, format))
        else:
            choice = random.randint(0, 2)
            if choice == 0:
                x = random.randint(1, 3999)
                examples.append(generate_zero_subtraction(x, 0, format))
            elif choice == 1:
                x = random.randint(1, 3999)
                examples.append(generate_zero_subtraction(x, x, format))
            else:
                examples.append(generate_zero_subtraction(0, 0, format))

    random.shuffle(examples)
    return examples


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_zero_accuracy(model, tokenizer, device, num_tests=100):
    """Evaluate model accuracy on zero operations

    Returns:
        Dictionary with accuracy for each zero operation type
    """
    results = {
        'n_plus_n': {'correct': 0, 'total': 0},
        'n_plus_x': {'correct': 0, 'total': 0},
        'x_plus_n': {'correct': 0, 'total': 0},
        'x_minus_n': {'correct': 0, 'total': 0},
        'x_minus_x': {'correct': 0, 'total': 0},
        'n_minus_n': {'correct': 0, 'total': 0}
    }

    tests_per_type = num_tests // 6

    # N + N = N
    for _ in range(tests_per_type):
        prompt = "N + N |"
        expected = "N"
        generated = generate_text(model, prompt, tokenizer, device)
        if expected in generated:
            results['n_plus_n']['correct'] += 1
        results['n_plus_n']['total'] += 1

    # N + X = X
    for _ in range(tests_per_type):
        x = random.randint(1, 100)
        x_roman = int_to_roman_subtractive(x)
        prompt = f"N + {x_roman} |"
        expected = x_roman
        generated = generate_text(model, prompt, tokenizer, device)
        if expected in generated:
            results['n_plus_x']['correct'] += 1
        results['n_plus_x']['total'] += 1

    # X + N = X
    for _ in range(tests_per_type):
        x = random.randint(1, 100)
        x_roman = int_to_roman_subtractive(x)
        prompt = f"{x_roman} + N |"
        expected = x_roman
        generated = generate_text(model, prompt, tokenizer, device)
        if expected in generated:
            results['x_plus_n']['correct'] += 1
        results['x_plus_n']['total'] += 1

    # X - N = X
    for _ in range(tests_per_type):
        x = random.randint(1, 100)
        x_roman = int_to_roman_subtractive(x)
        prompt = f"{x_roman} - N |"
        expected = x_roman
        generated = generate_text(model, prompt, tokenizer, device)
        if expected in generated:
            results['x_minus_n']['correct'] += 1
        results['x_minus_n']['total'] += 1

    # X - X = N
    for _ in range(tests_per_type):
        x = random.randint(1, 100)
        x_roman = int_to_roman_subtractive(x)
        prompt = f"{x_roman} - {x_roman} |"
        expected = "N"
        generated = generate_text(model, prompt, tokenizer, device)
        if expected in generated:
            results['x_minus_x']['correct'] += 1
        results['x_minus_x']['total'] += 1

    # N - N = N
    for _ in range(tests_per_type):
        prompt = "N - N |"
        expected = "N"
        generated = generate_text(model, prompt, tokenizer, device)
        if expected in generated:
            results['n_minus_n']['correct'] += 1
        results['n_minus_n']['total'] += 1

    # Calculate accuracies
    accuracies = {}
    for key, value in results.items():
        if value['total'] > 0:
            accuracies[key] = value['correct'] / value['total']
        else:
            accuracies[key] = 0.0

    overall = sum(v['correct'] for v in results.values()) / sum(v['total'] for v in results.values())
    accuracies['overall'] = overall

    return accuracies


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, tokenizer):
    """Train for one epoch"""
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


def save_lora_checkpoint(model, checkpoint_path, metadata):
    """Save only LoRA parameters"""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_state[name] = param.data.cpu()

    torch.save(lora_state, checkpoint_path)

    # Save metadata
    metadata_path = checkpoint_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LoRA Training for Zero Concept')
    parser.add_argument('--base-model', type=str, required=True,
                       help='Path to base model checkpoint')
    parser.add_argument('--format', type=str, choices=['simple', 'intermediate', 'full'],
                       default='intermediate', help='Training format complexity')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--examples', type=int, default=5000, help='Training examples')
    parser.add_argument('--checkpoint-dir', type=str, default='lora_checkpoints',
                       help='LoRA checkpoint directory')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("LORA TRAINING FOR ZERO CONCEPT")
    print("="*60)
    print(f"Using device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Format: {args.format}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print()

    # Load base model metadata
    base_path = Path(args.base_model)
    metadata_path = base_path.with_suffix('.json')

    if not base_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_path}")

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            base_metadata = json.load(f)
        print("Base model metadata:")
        print(json.dumps(base_metadata, indent=2))
        print()
    else:
        print("Warning: No metadata file found for base model")
        base_metadata = {}

    # Initialize tokenizer
    tokenizer = CharTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load base model
    print("Loading base model...")
    if 'model_config' in base_metadata:
        config = base_metadata['model_config']
        model = SimpleTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            max_seq_length=config.get('max_seq_length', 128),
            dropout=config.get('dropout', 0.1)
        )
    else:
        print("No model config in metadata, using defaults...")
        model = SimpleTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            nhead=8,
            num_layers=8,
            dim_feedforward=2048,
            max_seq_length=128
        )

    # Load base model weights
    model.load_state_dict(torch.load(args.base_model, map_location='cpu'))
    print("Base model loaded successfully")

    base_params = count_parameters(model, trainable_only=False)
    print(f"Base model parameters: {base_params:,}")

    # Apply LoRA
    print("\nApplying LoRA...")
    model = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )
    model = model.to(device)

    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    print(f"Trainable parameters (LoRA): {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Generate zero dataset
    print(f"Generating {args.examples} zero-concept training examples...")
    train_examples = generate_zero_dataset(args.examples, format=args.format)
    print(f"Generated {len(train_examples)} examples")
    print("\nSample examples:")
    for i in range(min(10, len(train_examples))):
        print(f"  {train_examples[i]}")
    print()

    # Create dataset
    dataset = RomanDataset(train_examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer (only LoRA parameters)
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    # Create metadata
    lora_metadata = {
        'base_model': str(base_path),
        'base_metadata': base_metadata,
        'lora_config': {
            'rank': args.lora_rank,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'trainable_params': trainable_params,
            'total_params': total_params
        },
        'training_config': {
            'format': args.format,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'num_examples': args.examples
        }
    }

    # Training loop
    print("="*60)
    print("TRAINING")
    print("="*60)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, optimizer, device, tokenizer)

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}: Loss = {loss:.4f}")

            # Evaluate zero accuracy
            zero_acc = evaluate_zero_accuracy(model, tokenizer, device, num_tests=60)
            print("Zero concept accuracy:")
            print(f"  N + N = N: {zero_acc['n_plus_n']:.2%}")
            print(f"  N + X = X: {zero_acc['n_plus_x']:.2%}")
            print(f"  X + N = X: {zero_acc['x_plus_n']:.2%}")
            print(f"  X - N = X: {zero_acc['x_minus_n']:.2%}")
            print(f"  X - X = N: {zero_acc['x_minus_x']:.2%}")
            print(f"  N - N = N: {zero_acc['n_minus_n']:.2%}")
            print(f"  Overall: {zero_acc['overall']:.2%}")

            # Show samples
            test_cases = [
                "N + N |",
                f"N + {int_to_roman_subtractive(random.randint(1, 50))} |",
                f"{int_to_roman_subtractive(random.randint(1, 50))} + N |",
                f"{int_to_roman_subtractive(random.randint(1, 50))} - N |",
            ]

            for prompt in test_cases:
                generated = generate_text(model, prompt, tokenizer, device)
                print(f"  {prompt} -> {generated}")

            # Save best model
            if loss < best_loss:
                best_loss = loss
                checkpoint_path = checkpoint_dir / "best_lora.pt"
                best_metadata = lora_metadata.copy()
                best_metadata['epoch'] = epoch + 1
                best_metadata['loss'] = loss
                best_metadata['zero_accuracy'] = zero_acc
                save_lora_checkpoint(model, checkpoint_path, best_metadata)
                print(f"New best loss! Saved to {checkpoint_path}")

        # Periodic checkpoints
        if (epoch + 1) % 50 == 0:
            checkpoint_path = checkpoint_dir / f"lora_epoch_{epoch+1}.pt"
            epoch_metadata = lora_metadata.copy()
            epoch_metadata['epoch'] = epoch + 1
            save_lora_checkpoint(model, checkpoint_path, epoch_metadata)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Final save
    final_path = checkpoint_dir / "final_lora.pt"
    final_metadata = lora_metadata.copy()
    final_metadata['training_complete'] = True
    save_lora_checkpoint(model, final_path, final_metadata)
    print(f"\nFinal LoRA saved: {final_path}")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    zero_acc = evaluate_zero_accuracy(model, tokenizer, device, num_tests=100)
    print("Zero concept accuracy (100 tests per type):")
    print(f"  N + N = N: {zero_acc['n_plus_n']:.2%}")
    print(f"  N + X = X: {zero_acc['n_plus_x']:.2%}")
    print(f"  X + N = X: {zero_acc['x_plus_n']:.2%}")
    print(f"  X - N = X: {zero_acc['x_minus_n']:.2%}")
    print(f"  X - X = N: {zero_acc['x_minus_x']:.2%}")
    print(f"  N - N = N: {zero_acc['n_minus_n']:.2%}")
    print(f"  Overall: {zero_acc['overall']:.2%}")


if __name__ == "__main__":
    main()
