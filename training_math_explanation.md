# Training Mathematics Explanation

## The Paper's Training Setup (Page 290)

The paper states:
> "In total, 768,000 instances were used in batches of 512, with training conducted over one hundred epochs, 1500 iterations per epoch, totalling 150,000 training steps."

## Your Understanding vs. Reality

### Your Understanding (Incorrect):
- Epochs: 101 (0...100)
- Batch size: 512
- Batches per epoch: 15
- Training instances: 12 × 512 × 101 = 620,544
- Validation instances: 155,136

### Paper's Actual Setup (Correct):
- **Total dataset size**: 768,000 instances
- **Batch size**: 512 instances per batch
- **Iterations per epoch**: 1,500
- **Epochs**: 100
- **Total training steps**: 150,000

## Developer's Explanation Breakdown

The developer is explaining fundamental deep learning concepts:

### 1. **What is a Batch?**
> "batch is the number of instances your training loop will see before doing a step and update the NN params"

**Meaning**: 
- A batch = 512 instances
- The model processes these 512 instances together
- After processing the batch, the model parameters (weights) are updated once

### 2. **What is an Iteration/Step?**
> "when you divide dataset size by the size of the batch, you get how many times the params will be updated in an epoch"

**Calculation**:
- Dataset size ÷ Batch size = Iterations per epoch
- 768,000 ÷ 512 = **1,500 iterations per epoch**

### 3. **What is an Epoch?**
> "It is the time it takes to go through the whole dataset one time."

**Meaning**:
- 1 epoch = processing all 768,000 instances once
- This requires 1,500 iterations (batches) to complete
- After 1,500 iterations, you've seen every instance once

## The Correct Mathematics

```
Total Dataset: 768,000 instances
Batch Size: 512 instances/batch
Iterations per Epoch: 768,000 ÷ 512 = 1,500 iterations
Epochs: 100
Total Training Steps: 1,500 × 100 = 150,000 steps
```

## Why Your Understanding Was Wrong

### Your Mistake:
You calculated **new instances per epoch** instead of understanding that epochs **reuse the same dataset**.

### The Reality:
- The **same 768,000 instances** are reused every epoch
- Each epoch processes all 768,000 instances in 1,500 batches
- Over 100 epochs, the model sees the same data 100 times

## Our Training vs. Paper's Training

### Our Training:
- **Total instances**: 7,680 (generated once)
- **Batch size**: 512
- **Batches per epoch**: 15
- **Iterations per epoch**: 15
- **Epochs**: 101
- **Total steps**: 15 × 101 = 1,515

### Paper's Training:
- **Total instances**: 768,000 (generated once)
- **Batch size**: 512  
- **Batches per epoch**: 1,500
- **Iterations per epoch**: 1,500
- **Epochs**: 100
- **Total steps**: 1,500 × 100 = 150,000

## Key Insights

1. **Dataset Reuse**: The same dataset is used repeatedly across epochs
2. **No New Generation**: No new instances are generated each epoch
3. **Parameter Updates**: Each batch triggers one parameter update
4. **Scale Difference**: Our training used **100x less data** than the paper

## Why This Matters

- **More data exposure**: The paper's model saw 100x more data
- **Better convergence**: More iterations lead to better learning
- **Longer training**: 150,000 steps vs our 1,515 steps
- **Better results**: More training typically yields better performance

This explains why the paper's training took 14+ hours while ours took 13 minutes.
