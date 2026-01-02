# Zero Test

The objective of the Zero Test is to narrow down whether or not a LoRA can add a completely new concept to an LLM.

## Mechanism

We will teach a model Roman numeral arithmetic. Once we have a model able to add and subtract arbitrary Roman numerals consistently, we will tune a LoRA for it, adding 'N' for the zero concept in with the arithmetic. We will then test the LoRA added to the model to see if it has included this concept into its training.

## Process

1. Determine an algorithmic process for Roman numeral arithmetic
2. Create a dataset using this process
3. Train a GPT model on the dataset until it is proficient
4. Add 'N' to the problems in a new dataset
5. Train a LoRA on the new dataset
6. Evaluate the model

## Roman Numerals

Roman numerals are represented as follows:

 - I is 1
 - V is 5
 - X is 10
 - L is 50
 - C is 100
 - D is 500
 - M is 1000

Roman numberals are combined in a left to right fashion as follows:

 - Repeated numerals are added together up to 3 times (II is 2)
 - Smaller numerals before larger are subtracted (IV is 4)
 - Smaller numerals after larger are added (VI is 6)
 
### Addition
 
To add two roman numerals, what you do is:

1. Convert any subtractive prefixes to additive suffixes. So, for example, IX would be rewritten to VIIII
2. Concatenate the two numbers to add
3. Sort the letters, large to small
4. Do internal sums (e.g., replace "IIIII" with "V")
5. Convert back to subtractive prefixes

So, for example: 123 + 69. In roman numerals, that's "CXXIII + "LXIX"

1. "CXXIII" has no subtractive prefixes. "LXIX" becomes "LXVIIII"
2. Concatenate: "CXXIIILXVIIII"
3. Sort: "CLXXXVIIIIIII"
4. Internal sum: reduce the "IIIIIII" to "VII" giving "CLXXXVVII"; then reduce the "VV" to "X": "CLXXXXII"
5. Switch to subtractive prefix: "XXXX" = "XL", giving "CLXLII". "LXL"="XC", giving "CXCII", or 192

[Credit](https://scienceblogs.com/goodmath/2006/08/16/roman-numerals-and-arithmetic)
 
### Subtraction

Subtraction isn't any harder than addition. To subtract A-B:

1. Convert subtractive prefixes to additive suffixes
2. Eliminate any common symbols that appear in both A and B
3. For the largest remaining symbol in B, take the first symbol in A larger than it, and expand it. Then go back to step two, until there's nothing left
4. Convert back to subtractive prefixes

So 192-69 = "CXCII-LXIX"

1. Remove prefixes: CLXXXXII - LXVIIII
2. Remove common symbols. CXXX - VII
3. Expand an "X" in "CXXX": CXXVIIIII - VII
4. Remove common symbols: CXXIII = 123

[Credit](https://scienceblogs.com/goodmath/2006/08/16/roman-numerals-and-arithmetic)


