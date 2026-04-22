# Overleaf Report Guide

To complete your final report, follow these steps:

1. **Create a new Project on Overleaf:**
   - Go to [Overleaf](https://www.overleaf.com/).
   - Click "New Project" -> "Blank Project".

2. **Upload the LaTeX source:**
   - Copy the contents of `overleaf/main.tex` and paste it into the `main.tex` file in your Overleaf project.

3. **Upload the Figures:**
   - You need to upload the generated plots into the Overleaf project root so they can be rendered directly by `main.tex`.
   - **Upload these files:**
     - `fig_arch_comparison.png`
     - `fig_bottleneck.png`
     - `fig_generalization.png`
     - `fig_snr_comparison.png`
     - `reconstructions.png` (You can add this to show example denoisings!)

4. **Add Reconstruction Examples:**
   - I have included placeholders for the main comparison charts. You should also add a section for "Qualitative Results" using `reconstructions.png` to show side-by-side examples of clean vs. noisy vs. reconstructed signals.

5. **Update the Results Table:**
   - Open `experiments/results.csv`.
   - Update the numbers in the `\begin{tabular}` section of `main.tex` with your actual best scores.

### Recommended Diagrams to Include:
1. **Model Architecture Diagram:** (Optional but good) A block diagram showing the Encoder -> Bottleneck -> Decoder flow.
2. **Architecture Comparison:** (`fig_arch_comparison.png`) This is the "headline" chart.
3. **SNR Improvement:** (`fig_snr_comparison.png`) Proves your model actually helps.
4. **Generalization Curve:** (`fig_generalization.png`) Shows how robust the model is to different noise levels.
5. **Visual Examples:** (`reconstructions.png`) Essential for the "Visualizations" deliverable in the brief.
