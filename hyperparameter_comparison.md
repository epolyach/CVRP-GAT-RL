# Hyperparameter Comparison: Our Training vs. Paper (131659.pdf)

This document provides a detailed comparison between the hyperparameters used in our training run and those specified in the research paper `131659.pdf`.

---

## Comparison Table

| Hyperparameter | Our Training (`hyperparameters_summary.txt`) | Paper (`131659.pdf`) | Match? | Notes |
|:---|:---|:---|:---:|:---|
| **Problem Setup** | | | | |
| Nodes (N) | 20 | 21 (customers + depot) | ✅ | Same. The paper is more explicit. |
| **Training** | | | | |
| Batch Size | 512 | 512 | ✅ | |
| Batches/Epoch | 15 | 1500 iterations/epoch | ⚠️ | **Major Discrepancy**. Our training is much shorter. |
| Total Training Instances | 7,680 | 768,000 | ⚠️ | **Major Discrepancy**. The paper uses 100x more data. |
| **Model Architecture** | | | | |
| Input Vertex Dimension | 4 | 3 | ⚠️ | **Major Discrepancy**. Our model has an extra feature due to a code fix. |
| Input Edge Dimension | 1 | 1 | ✅ | |
| Vertex Embedding Dimension | 128 | 128 | ✅ | |
| Edge Embedding Dimension | 16 | 16 | ✅ | |
| Layers in Encoder | 4 | 4 | ✅ | |
| Dropout Rate | 0.1 | 0.6 | ⚠️ | **Major Discrepancy**. Our model has much less regularization. |
| **Optimization** | | | | |
| Learning Rate (LR) | 1e-4 | 1e-4 | ✅ | |
| SOFTMAX Temperature (T) | 2.5 | 2.5 | ✅ | |
| **Training Time** | | | | |
| Training Time (DiCE) | 13.4 minutes | 14.24 hours | ⚠️ | Consistent with the discrepancy in training data size. |
| Training Time (Greedy Rollout) | Not run | 20.52 hours | N/A | |

---

## Summary of Key Differences

1.  **Training Data**: The paper uses **100 times more training data** than our run (768,000 instances vs. 7,680). This is the most significant difference and explains the much longer training time in the paper.

2.  **Input Features**: Our model uses an extra input feature (4 dimensions vs. 3). This was a necessary fix to get the provided code to run.

3.  **Dropout Rate**: The paper uses a much higher dropout rate (0.6 vs. 0.1). This suggests the paper's model was regularized more heavily, which is common with larger datasets.

4.  **Training Time**: The training times are consistent with the amount of data used. Our shorter training time is a direct result of using less data.

---

## Conclusion

Our training run is a **scaled-down version** of the one described in the paper. While many of the core architectural hyperparameters match, the **training regime is substantially different**.

To more closely replicate the paper's results, the following changes would be required:
*   Increase the number of training instances to 768,000.
*   Increase `batches_per_epoch` to 1500.
*   Increase the dropout rate to 0.6.
*   Investigate and fix the input dimension issue to use 3 features instead of 4.

The current results are valuable as a baseline but are not directly comparable to the paper's due to these significant differences.
