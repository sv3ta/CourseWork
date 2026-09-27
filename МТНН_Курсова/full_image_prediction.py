import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import os
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

def predict_full_image(rf_model, ema_proba, X_full):
    """Predict labels for the entire image using the trained model."""
    print("Predicting on full image (this may take a while)...")
    

    height, width, n_bands = X_full.shape
    X_2d = X_full.reshape(-1, n_bands)
    

    batch_size = 10000
    n_samples = X_2d.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    
    full_proba = np.zeros((n_samples, rf_model.n_classes_))
    
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        X_batch = X_2d[start_idx:end_idx]
        full_proba[start_idx:end_idx] = rf_model.predict_proba(X_batch)
    

    full_preds = np.argmax(full_proba, axis=1)
    

    pred_img = full_preds.reshape(height, width)
    
    return pred_img, full_proba

def create_colormap(n_classes):
    """Create a colormap for visualization."""
    colors = ['black', 'green', 'blue', 'red', 'cyan', 'magenta', 'yellow', 
              'orange', 'purple', 'brown', 'pink', 'lime', 'teal', 'coral']
    

    if n_classes + 1 > len(colors):
        from matplotlib.colors import hsv_to_rgb
        hsv_colors = [(i/n_classes, 1.0, 0.9) for i in range(n_classes)]
        additional_colors = [tuple(hsv_to_rgb(c)) for c in hsv_colors]
        colors = colors + additional_colors
    
    return ListedColormap(colors[:n_classes+1])

def show_rgb_composite(X_raw, title, save_path=None):
    """Show RGB composite of hyperspectral image."""
    rgb = X_raw[:, :, [30, 20, 10]]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(title, fontsize=14)
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def show_label_map(pred_map, title, class_names, save_path=None):
    """Show label map with legend."""

    n_classes = len(class_names)
    cmap = create_colormap(n_classes)
    bounds = np.arange(n_classes+2) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(10, 8))
    img = plt.imshow(pred_map, cmap=cmap, norm=norm)
    plt.title(title, fontsize=14)
    plt.axis("off")
    

    legend_elements = [Patch(facecolor=cmap(0), edgecolor='k', label='Background')]
    for i, name in enumerate(class_names):
        legend_elements.append(Patch(facecolor=cmap(i+1), edgecolor='k', label=name))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
               borderaxespad=0., fontsize='small')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def compare_ground_truth_predictions(gt_map, pred_map, class_names, save_path=None):
    """Compare ground truth with predictions side by side."""
    n_classes = len(class_names)
    cmap = create_colormap(n_classes)
    bounds = np.arange(n_classes+2) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(15, 8))
    

    plt.subplot(1, 2, 1)
    plt.imshow(gt_map, cmap=cmap, norm=norm)
    plt.title("Ground Truth", fontsize=14)
    plt.axis("off")
    

    plt.subplot(1, 2, 2)
    plt.imshow(pred_map, cmap=cmap, norm=norm)
    plt.title("Model Predictions", fontsize=14)
    plt.axis("off")
    

    legend_elements = [Patch(facecolor=cmap(0), edgecolor='k', label='Background')]
    for i, name in enumerate(class_names):
        legend_elements.append(Patch(facecolor=cmap(i+1), edgecolor='k', label=name))
    
    plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0), 
                 ncol=7, fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def calculate_accuracy_metrics(gt_map, pred_map, class_names):
    """Calculate accuracy metrics for valid pixels (non-background)."""

    gt_flat = gt_map.flatten()
    pred_flat = pred_map.flatten()
    

    valid_mask = gt_flat > 0
    gt_valid = gt_flat[valid_mask] - 1
    pred_valid = pred_flat[valid_mask]
    

    accuracy = accuracy_score(gt_valid, pred_valid)
    

    cm = confusion_matrix(gt_valid, pred_valid)
    report = classification_report(gt_valid, pred_valid, target_names=class_names, digits=3)
    
    return accuracy, cm, report

def plot_confidence_map(confidence, title, save_path=None):
    """Plot prediction confidence map."""
    plt.figure(figsize=(10, 8))
    plt.imshow(confidence, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Confidence')
    plt.title(title, fontsize=14)
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_error_map(gt_map, pred_map, class_names, save_path=None):
    """Create an error map highlighting correct and incorrect predictions."""

    error_map = (gt_map == pred_map).astype(int)
    

    error_map[gt_map == 0] = 2
    

    cmap = ListedColormap(['red', 'green', 'black'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(error_map, cmap=cmap, norm=norm)
    plt.title("Error Map", fontsize=14)
    

    legend_elements = [
        Patch(facecolor='red', edgecolor='k', label='Incorrect'),
        Patch(facecolor='green', edgecolor='k', label='Correct'),
        Patch(facecolor='black', edgecolor='k', label='Background')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def main():

    class_names = [
        "Scrub",
        "Willow swamp",
        "CP hammock",
        "Slash pine",
        "Oak/Broadleaf",
        "Hardwood swamp",
        "Graminoid marsh",
        "Spartina marsh",
        "Cattail marsh",
        "Salt marsh",
        "Mud flats",
        "Water",
        "Cabbage palm"
    ]

    model_path = 'assrf_tod_model.pkl'
    output_dir = 'Image_pred'
    
    print(f"Loading model from: {model_path}")
    

    model_data = joblib.load(model_path)
    rf_model = model_data['rf_model']
    ema_proba = model_data.get('ema_proba', None)
    
    try:

        X_full = loadmat('data/KSC.mat')['KSC']
        y_full = loadmat('data/KSC_gt.mat')['KSC_gt']

        print("Loaded hyperspectral data successfully")
        

        gt_map = y_full
        

        pred_map, full_proba = predict_full_image(rf_model, ema_proba, X_full)
        

        confidence_map = np.max(full_proba, axis=1).reshape(X_full.shape[0], X_full.shape[1])
        

        print("Generating visualizations...")
        

        rgb_fig = show_rgb_composite(X_full, "Kennedy Space Center - RGB Composite", 
                                    os.path.join(output_dir, 'full_rgb_composite.png'))
        

        gt_fig = show_label_map(gt_map, "Ground Truth Labels", class_names,
                              os.path.join(output_dir, 'full_ground_truth.png'))
        

        pred_fig = show_label_map(pred_map, "Full Image Predictions", class_names,
                                os.path.join(output_dir, 'full_predictions.png'))
        

        compare_fig = compare_ground_truth_predictions(gt_map, pred_map, class_names,
                                                    os.path.join(output_dir, 'gt_vs_pred_comparison.png'))
        

        conf_fig = plot_confidence_map(confidence_map, "Prediction Confidence",
                                     os.path.join(output_dir, 'full_confidence_map.png'))
        

        error_fig = plot_error_map(gt_map, pred_map, class_names,
                                 os.path.join(output_dir, 'error_map.png'))
        

        print("Calculating accuracy metrics...")
        accuracy, cm, report = calculate_accuracy_metrics(gt_map, pred_map, class_names)
        
        print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(report)
        

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (All Valid Pixels)")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'full_confusion_matrix.png'), dpi=300)
        

        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(class_names)), class_accuracies * 100)
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Class Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=300)
        

        with open(os.path.join(output_dir, 'full_prediction_results.txt'), 'w') as f:
            f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nPer-Class Accuracy:\n")
            for i, acc in enumerate(class_accuracies):
                f.write(f"{class_names[i]}: {acc * 100:.2f}%\n")
        
        print(f"All visualizations and results saved to {output_dir}")
        
        plt.show()
        
    except FileNotFoundError:
        print("Error: Could not find X_full.npy or y_full.npy. Make sure these files exist in the current directory.")
        return

if __name__ == "__main__":
    main()