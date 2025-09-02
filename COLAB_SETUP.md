# 🚀 Google Colab Setup Guide

## Quick Start

1. **Go to Google Colab**: [colab.research.google.com](https://colab.research.google.com)

2. **Upload the notebook**: 
   - Click "Upload" 
   - Select `colab_training.ipynb` from your local files

3. **Enable GPU**:
   - Go to `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Click `Save`

4. **Run the notebook**:
   - Click `Runtime` → `Run all`
   - Or run each cell individually with `Shift + Enter`

## What You'll See

### Real-time Visualizations:
- **📊 Loss Graphs**: Live training and test loss curves
- **🎨 Generated Samples**: 16 new samples after each epoch
- **🔄 Denoising Process**: Side-by-side comparison of original → noisy → denoised
- **📈 Training Progress**: Real-time updates with progress bars

### Output Files:
- `colab_outputs/samples/`: Sample images from each epoch
- `colab_outputs/checkpoints/`: Model checkpoints every 10 epochs
- `colab_outputs/final_samples_64.png`: 64 final generated samples
- `colab_outputs/training_summary.png`: Training curves and summary

## Expected Training Time

- **GPU (T4)**: ~2-3 hours for 50 epochs
- **CPU**: ~8-12 hours for 50 epochs

## Tips

1. **Monitor Progress**: The notebook updates in real-time, so you can see progress without waiting
2. **Save Checkpoints**: Model saves every 10 epochs automatically
3. **Download Results**: All outputs are automatically zipped and downloaded at the end
4. **Interrupt Safely**: You can stop training anytime - checkpoints are saved

## Troubleshooting

- **No GPU**: Make sure you selected GPU in runtime settings
- **Out of Memory**: Reduce batch size in the training cell
- **Slow Training**: Check if GPU is actually being used (should show CUDA device)

## Repository

Your code is available at: https://github.com/yatuzhang/ddpm-mnist

Happy training! 🎉
