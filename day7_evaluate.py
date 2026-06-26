"""
Day 7: Evaluate the trained face mask detection model
- Generate classification report
- Plot confusion matrix
- Run inference on random test images
- Display misclassified examples
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow.keras.models import load_model


def load_data():
    """Load test dataset"""
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    
    print("Loading test data...")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    # Normalize test data
    
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_test, y_test


def load_trained_model(model_path):
    """Load the trained model"""
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    return model


def get_class_names():
    return ["without_mask", "with_mask"]


def evaluate_model(model, X_test, y_test):
    """Generate predictions and evaluation metrics"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Convert y_test to class indices if it's one-hot encoded
    if len(y_test.shape) > 1:
        y_test_classes = np.argmax(y_test, axis=1)
    else:
        y_test_classes = y_test
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_classes, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    class_names = get_class_names()
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(
        y_test_classes, y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    return y_pred, y_pred_probs, y_test_classes


def plot_confusion_matrix(y_test, y_pred):
    """Plot and display confusion matrix"""
    print("\nGenerating confusion matrix...")
    class_names = get_class_names()
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Face Mask Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()
    
    return cm


def show_random_predictions(X_test, y_test, y_pred, y_pred_probs, num_samples=10):
    """Display random test images with predictions"""
    print("\n" + "="*60)
    print(f"INFERENCE ON {num_samples} RANDOM TEST IMAGES")
    print("="*60)
    
    class_names = get_class_names()
    
    # Select random indices
    random_indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, img_idx in enumerate(random_indices):
        ax = axes[idx]
        image = X_test[img_idx]
        true_label = y_test[img_idx]
        pred_label = y_pred[img_idx]
        confidence = y_pred_probs[img_idx][pred_label]
        
        # Display image
        ax.imshow(image)
        
        # Color code: green if correct, red if incorrect
        color = 'green' if true_label == pred_label else 'red'
        
        title = (f"True: {class_names[true_label]}\n"
                f"Pred: {class_names[pred_label]} ({confidence:.2%})")
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.axis('off')
    
    plt.suptitle('Random Test Image Predictions (Green=Correct, Red=Incorrect)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('random_predictions.png', dpi=150, bbox_inches='tight')
    print("Random predictions saved as 'random_predictions.png'")
    plt.show()


def find_misclassified(X_test, y_test, y_pred, y_pred_probs):
    """Find and display misclassified examples"""
    print("\n" + "="*60)
    print("MISCLASSIFIED EXAMPLES")
    print("="*60)
    
    class_names = get_class_names()
    
    # Find misclassified indices
    misclassified_mask = y_test != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    num_misclassified = len(misclassified_indices)
    print(f"\nTotal misclassified: {num_misclassified} out of {len(y_test)} "
          f"({num_misclassified/len(y_test)*100:.2f}%)")
    
    if num_misclassified == 0:
        print("No misclassified examples found! Perfect predictions!")
        return
    
    # Show up to 8 misclassified examples
    num_to_show = min(8, num_misclassified)
    selected_indices = misclassified_indices[:num_to_show]
    
    print(f"\nDisplaying first {num_to_show} misclassified examples:")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, img_idx in enumerate(selected_indices):
        ax = axes[idx]
        image = X_test[img_idx]
        true_label = y_test[img_idx]
        pred_label = y_pred[img_idx]
        confidence = y_pred_probs[img_idx][pred_label]
        
        ax.imshow(image)
        
        title = (f"True: {class_names[true_label]}\n"
                f"Pred: {class_names[pred_label]} ({confidence:.2%})")
        ax.set_title(title, fontsize=10, fontweight='bold', color='red')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_to_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Misclassified Examples (Model Predictions)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=150, bbox_inches='tight')
    print(f"Misclassified examples saved as 'misclassified_examples.png'")
    plt.show()
    
    # Print detailed misclassification statistics
    print("\n" + "-"*60)
    print("MISCLASSIFICATION BREAKDOWN:")
    print("-"*60)
    
    for true_class in range(len(class_names)):
        for pred_class in range(len(class_names)):
            if true_class != pred_class:
                mask = (y_test == true_class) & (y_pred == pred_class)
                count = np.sum(mask)
                if count > 0:
                    print(f"  {class_names[true_class]} misclassified as {class_names[pred_class]}: {count}")


def print_detailed_statistics(y_test, y_pred, y_pred_probs):
    """Print additional detailed statistics"""
    print("\n" + "="*60)
    print("DETAILED STATISTICS")
    print("="*60)
    
    class_names = get_class_names()
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = y_test == class_idx
        class_count = np.sum(class_mask)
        
        # Predictions for this class
        class_preds = y_pred[class_mask]
        correct = np.sum(class_preds == class_idx)
        
        # Average confidence for this class
        class_confidence = y_pred_probs[class_mask, class_idx]
        avg_confidence = np.mean(class_confidence)
        
        print(f"\n{class_name}:")
        print(f"  Total samples: {class_count}")
        print(f"  Correctly classified: {correct} ({correct/class_count*100:.2f}%)")
        print(f"  Average confidence: {avg_confidence:.4f}")
        print(f"  Min confidence: {np.min(class_confidence):.4f}")
        print(f"  Max confidence: {np.max(class_confidence):.4f}")
        print(f"  Std confidence: {np.std(class_confidence):.4f}")


def main():
    """Main evaluation pipeline"""
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "model" / "mask_detector.h5"
    
    # Load data and model
    X_test, y_test = load_data()
    model = load_trained_model(model_path)
    
    # Evaluate model
    y_pred, y_pred_probs, y_test_classes = evaluate_model(model, X_test, y_test)
    
    # Generate visualizations
    plot_confusion_matrix(y_test_classes, y_pred)
    show_random_predictions(X_test, y_test_classes, y_pred, y_pred_probs, num_samples=10)
    find_misclassified(X_test, y_test_classes, y_pred, y_pred_probs)
    
    # Print detailed statistics
    print_detailed_statistics(y_test_classes, y_pred, y_pred_probs)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - random_predictions.png")
    print("  - misclassified_examples.png")


if __name__ == '__main__':
    main()
