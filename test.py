from ultralytics import YOLO
import os
import torch
import glob

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'data.yaml')

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on device: {device}")

    # Find all .pt files in the current directory
    found_pt_files = glob.glob(os.path.join(current_dir, '*.pt'))
    models_to_test_paths = {os.path.basename(f) for f in found_pt_files} # Use a set to avoid duplicates

    # Explicitly add 'best_obstacle_model.pt' and 'last.pt' if they exist,
    # as these are the names used by train.py for the best and last models.
    if os.path.exists(os.path.join(current_dir, 'best_obstacle_model.pt')):
        models_to_test_paths.add('best_obstacle_model.pt')
    if os.path.exists(os.path.join(current_dir, 'last.pt')):
        models_to_test_paths.add('last.pt')
    
    models_to_compare = sorted(list(models_to_test_paths))

    print(f"Models to test: {models_to_compare}")
    
    summary_results = []

    for model_name in models_to_compare:
        model_path = os.path.join(current_dir, model_name)

        if not os.path.exists(model_path):
            print(f"Model not found: {model_name}")
            continue

        print(f"\n{'='*20} Evaluating {model_name} {'='*20}")
        try:
            model = YOLO(model_path) # verbose=True enables per-class accuracy reporting
            # verbose=True enables per-class accuracy reporting
            metrics = model.val(data=data_path, split='test', device=device, verbose=True)
            
            summary_results.append({
                'Model': model_name,
                'mAP': metrics.box.map,
                'Precision': metrics.box.mp,
                'Recall': metrics.box.mr
            })
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    print(f"\n{'='*60}")
    print(f"{'FINAL COMPARISON':^60}")
    print(f"{'='*60}")
    print(f"{'Model':<30} | {'mAP@0.5:0.95':<12} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 68)
    summary_results.sort(key=lambda x: x['mAP'], reverse=True)

    for res in summary_results:
        print(f"{res['Model']:<30} | {res['mAP']:<12.4f} | {res['Precision']:<10.4f} | {res['Recall']:<10.4f}")