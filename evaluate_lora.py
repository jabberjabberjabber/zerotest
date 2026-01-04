"""
LoRA Evaluation Script - Zero Concept Testing

This script evaluates and compares:
  1. Base model performance on standard Roman numerals
  2. Base model performance on zero concepts (should fail)
  3. LoRA model performance on standard Roman numerals (should maintain)
  4. LoRA model performance on zero concepts (should succeed)

The goal is to verify that LoRA can add a completely new concept (zero)
without degrading the base model's existing knowledge.

Usage:
  # Evaluate base model only
  python evaluate_lora.py --base-model checkpoints/best_model.pt

  # Compare base model vs LoRA model
  python evaluate_lora.py \\
    --base-model checkpoints/best_model.pt \\
    --lora lora_checkpoints/best_lora.pt \\
    --num-tests 100

  # Interactive mode - test prompts in real-time
  python evaluate_lora.py \\
    --base-model checkpoints/best_model.pt \\
    --lora lora_checkpoints/best_lora.pt \\
    --interactive

Output:
  - Accuracy metrics for all operation types
  - Sample generations showing correct/incorrect predictions
  - Comparison summary and success determination

Interactive Mode:
  - Enter prompts directly and see results
  - Switch between base and LoRA models with 'base' and 'lora' commands
  - Type 'quit' or 'exit' to end session

Expected Results:
  - Base: ~97% on standard, ~0% on zero
  - LoRA: ~95% on standard, ~95% on zero (SUCCESS!)
"""

import torch
import argparse
import json
from pathlib import Path
from roman_numeral_trainer import (
    CharTokenizer,
    SimpleTransformer,
    generate_text,
    evaluate_accuracy,
    evaluate_subtraction_accuracy
)
from roman_numeral_lora_trainer import (
    apply_lora_to_model,
    evaluate_zero_accuracy
)

def interactive_mode(base_model, lora_model, tokenizer, device, has_lora=False):
    """Interactive prompting mode for testing base and LoRA models"""
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nEnter Roman numeral arithmetic prompts.")
    print("\nExamples for standard operations:")
    print("  X + V |")
    print("  XX - III |")
    print("  C + L |")

    if has_lora:
        print("\nExamples for zero concept (with LoRA):")
        print("  N + N |")
        print("  N + X |")
        print("  X + N |")
        print("  X - N |")
        print("  V - V |")

    print("\nCommands:")
    print("  'base' - Switch to base model")
    if has_lora:
        print("  'lora' - Switch to LoRA model")
    print("  'quit' or 'exit' - End session")
    print()

    # Start with LoRA model if available, otherwise base
    current_model = lora_model if has_lora else base_model
    current_name = "LoRA" if has_lora else "Base"

    print(f"Current model: {current_name}")
    print()

    while True:
        try:
            prompt = input(f"[{current_name}] Prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break

            if prompt.lower() == 'base':
                current_model = base_model
                current_name = "Base"
                print(f"Switched to {current_name} model\n")
                continue

            if has_lora and prompt.lower() == 'lora':
                current_model = lora_model
                current_name = "LoRA"
                print(f"Switched to {current_name} model\n")
                continue

            if not prompt:
                continue

            # Add pipe separator if not present
            if not prompt.endswith('|'):
                prompt = prompt + ' |'

            # Generate response
            print("Generating...", end='', flush=True)
            generated = generate_text(current_model, prompt, tokenizer, device, max_new_tokens=300)
            print("\r" + " " * 20 + "\r", end='')  # Clear the "Generating..." message

            print(f"Generated: {generated}")
            print()

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print()


def load_model_with_lora(base_model_path, lora_path):
    """Load base model and apply LoRA weights"""
    # Load base model metadata
    base_metadata_path = Path(base_model_path).with_suffix('.json')
    if base_metadata_path.exists():
        with open(base_metadata_path, 'r') as f:
            base_metadata = json.load(f)
    else:
        base_metadata = {}

    # Initialize tokenizer
    tokenizer = CharTokenizer()

    # Create base model
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
        model = SimpleTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            nhead=8,
            num_layers=8,
            dim_feedforward=2048,
            max_seq_length=128
        )

    # Load base weights
    model.load_state_dict(torch.load(base_model_path, map_location='cpu'))

    # Load LoRA metadata
    lora_metadata_path = Path(lora_path).with_suffix('.json')
    if lora_metadata_path.exists():
        with open(lora_metadata_path, 'r') as f:
            lora_metadata = json.load(f)
    else:
        lora_metadata = {'lora_config': {'rank': 8, 'alpha': 16, 'dropout': 0.0}}

    # Apply LoRA structure
    lora_config = lora_metadata.get('lora_config', {})
    model = apply_lora_to_model(
        model,
        rank=lora_config.get('rank', 8),
        alpha=lora_config.get('alpha', 16),
        dropout=lora_config.get('dropout', 0.0)
    )

    # Load LoRA weights
    lora_state = torch.load(lora_path, map_location='cpu')

    # Load only LoRA parameters
    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name] = param
    model.load_state_dict(model_state)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Evaluate Base Model vs LoRA Model')
    parser.add_argument('--base-model', type=str, required=True,
                       help='Path to base model checkpoint')
    parser.add_argument('--lora', type=str, default=None,
                       help='Path to LoRA checkpoint (optional)')
    parser.add_argument('--num-tests', type=int, default=100,
                       help='Number of tests per category')
    parser.add_argument('--interactive', action='store_true',
                       help='Enter interactive mode for testing prompts')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)

    # Load base model
    print(f"\nLoading base model from: {args.base_model}")
    base_metadata_path = Path(args.base_model).with_suffix('.json')
    if base_metadata_path.exists():
        with open(base_metadata_path, 'r') as f:
            base_metadata = json.load(f)
    else:
        base_metadata = {}

    tokenizer = CharTokenizer()

    if 'model_config' in base_metadata:
        config = base_metadata['model_config']
        base_model = SimpleTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            max_seq_length=config.get('max_seq_length', 128),
            dropout=config.get('dropout', 0.1)
        )
    else:
        base_model = SimpleTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            nhead=8,
            num_layers=8,
            dim_feedforward=2048,
            max_seq_length=128
        )

    base_model.load_state_dict(torch.load(args.base_model, map_location='cpu'))
    base_model = base_model.to(device)
    base_model.eval()

    print("Base model loaded successfully")

    # Enter interactive mode if requested
    if args.interactive:
        lora_model = None
        if args.lora:
            print(f"\nLoading LoRA from: {args.lora}")
            lora_model, _ = load_model_with_lora(args.base_model, args.lora)
            lora_model = lora_model.to(device)
            lora_model.eval()
            print("LoRA model loaded successfully")

        interactive_mode(base_model, lora_model, tokenizer, device, has_lora=(args.lora is not None))
        return

    # Evaluate base model
    print("\n" + "="*70)
    print("BASE MODEL EVALUATION (Standard Roman Numerals)")
    print("="*70)

    print(f"\nTesting on {args.num_tests} examples per operation...")
    add_acc = evaluate_accuracy(base_model, tokenizer, device, num_tests=args.num_tests,
                                 min_val=1, max_val=100)
    sub_acc = evaluate_subtraction_accuracy(base_model, tokenizer, device,
                                            num_tests=args.num_tests, min_val=1, max_val=100)

    print(f"Addition accuracy: {add_acc:.2%}")
    print(f"Subtraction accuracy: {sub_acc:.2%}")
    print(f"Overall accuracy: {(add_acc + sub_acc) / 2:.2%}")

    # Test zero concept on base model
    print("\n" + "="*70)
    print("BASE MODEL EVALUATION (Zero Concept - Should Fail)")
    print("="*70)

    zero_acc_base = evaluate_zero_accuracy(base_model, tokenizer, device,
                                           num_tests=args.num_tests)
    print(f"\nZero concept accuracy:")
    print(f"  N + N = N: {zero_acc_base['n_plus_n']:.2%}")
    print(f"  N + X = X: {zero_acc_base['n_plus_x']:.2%}")
    print(f"  X + N = X: {zero_acc_base['x_plus_n']:.2%}")
    print(f"  X - N = X: {zero_acc_base['x_minus_n']:.2%}")
    print(f"  X - X = N: {zero_acc_base['x_minus_x']:.2%}")
    print(f"  N - N = N: {zero_acc_base['n_minus_n']:.2%}")
    print(f"  Overall: {zero_acc_base['overall']:.2%}")

    print("\nSample outputs from base model (should be wrong):")
    import random
    from roman_numeral_trainer import int_to_roman_subtractive

    test_prompts = [
        ("N + N |", "N"),
        (f"N + {int_to_roman_subtractive(25)} |", int_to_roman_subtractive(25)),
        (f"{int_to_roman_subtractive(42)} + N |", int_to_roman_subtractive(42)),
        (f"{int_to_roman_subtractive(15)} - N |", int_to_roman_subtractive(15)),
        (f"{int_to_roman_subtractive(33)} - {int_to_roman_subtractive(33)} |", "N"),
    ]

    for prompt, expected in test_prompts:
        generated = generate_text(base_model, prompt, tokenizer, device)
        status = "✓" if expected in generated else "✗"
        print(f"  {status} {prompt} -> {generated} (expected: {expected})")

    # Evaluate LoRA model if provided
    if args.lora:
        print("\n" + "="*70)
        print("LORA MODEL EVALUATION")
        print("="*70)

        print(f"\nLoading LoRA from: {args.lora}")
        lora_model, _ = load_model_with_lora(args.base_model, args.lora)
        lora_model = lora_model.to(device)
        lora_model.eval()
        print("LoRA model loaded successfully")

        # Test standard operations (should maintain performance)
        print("\n" + "-"*70)
        print("Standard Roman Numerals (should maintain base accuracy)")
        print("-"*70)

        add_acc_lora = evaluate_accuracy(lora_model, tokenizer, device,
                                         num_tests=args.num_tests, min_val=1, max_val=100)
        sub_acc_lora = evaluate_subtraction_accuracy(lora_model, tokenizer, device,
                                                     num_tests=args.num_tests, min_val=1, max_val=100)

        print(f"Addition accuracy: {add_acc_lora:.2%}")
        print(f"Subtraction accuracy: {sub_acc_lora:.2%}")
        print(f"Overall accuracy: {(add_acc_lora + sub_acc_lora) / 2:.2%}")

        # Test zero concept (should work now!)
        print("\n" + "-"*70)
        print("Zero Concept (should work with LoRA)")
        print("-"*70)

        zero_acc_lora = evaluate_zero_accuracy(lora_model, tokenizer, device,
                                               num_tests=args.num_tests)
        print(f"\nZero concept accuracy:")
        print(f"  N + N = N: {zero_acc_lora['n_plus_n']:.2%}")
        print(f"  N + X = X: {zero_acc_lora['n_plus_x']:.2%}")
        print(f"  X + N = X: {zero_acc_lora['x_plus_n']:.2%}")
        print(f"  X - N = X: {zero_acc_lora['x_minus_n']:.2%}")
        print(f"  X - X = N: {zero_acc_lora['x_minus_x']:.2%}")
        print(f"  N - N = N: {zero_acc_lora['n_minus_n']:.2%}")
        print(f"  Overall: {zero_acc_lora['overall']:.2%}")

        print("\nSample outputs from LoRA model (should be correct):")
        for prompt, expected in test_prompts:
            generated = generate_text(lora_model, prompt, tokenizer, device)
            status = "✓" if expected in generated else "✗"
            print(f"  {status} {prompt} -> {generated} (expected: {expected})")

        # Comparison summary
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        print("\nStandard Operations:")
        print(f"  Base model:  {(add_acc + sub_acc) / 2:.2%}")
        print(f"  LoRA model:  {(add_acc_lora + sub_acc_lora) / 2:.2%}")
        print(f"  Difference:  {((add_acc_lora + sub_acc_lora) / 2 - (add_acc + sub_acc) / 2):.2%}")

        print("\nZero Concept:")
        print(f"  Base model:  {zero_acc_base['overall']:.2%}")
        print(f"  LoRA model:  {zero_acc_lora['overall']:.2%}")
        print(f"  Improvement: {(zero_acc_lora['overall'] - zero_acc_base['overall']):.2%}")

        print("\n" + "="*70)
        if zero_acc_lora['overall'] > 0.9 and (add_acc_lora + sub_acc_lora) / 2 > 0.9:
            print("SUCCESS! LoRA successfully added zero concept while maintaining base skills")
        elif zero_acc_lora['overall'] > 0.9:
            print("PARTIAL SUCCESS! LoRA learned zero but may have affected base performance")
        else:
            print("NEEDS MORE TRAINING! LoRA has not fully learned the zero concept")
        print("="*70)


if __name__ == "__main__":
    main()
