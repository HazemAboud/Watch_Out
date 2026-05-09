from ultralytics import YOLO
import os
import torch
import shutil
import gc
import glob

if __name__ == '__main__':
    hyperparams_grid = [
        { 'model': 'yolo11s.pt', 'optimizer': 'AdamW', 'lr0': 0.001, 'batch': 8, 'imgsz': 960, 'epochs': 100},
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'data.yaml')

    print("Clearing dataset cache files...")
    for cache_file in glob.glob(os.path.join(os.path.dirname(data_path), '**', '*.cache'), recursive=True):
        try:
            os.remove(cache_file)
            print(f"Deleted: {cache_file}")
        except Exception as e:
            print(f"Error deleting {cache_file}: {e}")

    best_map = 0.0
    best_model_path = None
    best_params = None

    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 0:
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Training on device: {device} ({torch.cuda.get_device_name(0)}) | VRAM: {vram:.2f} GB")
    else:
        print("Training on device: CPU")
        print(f"PyTorch Version: {torch.__version__}")

    for params in hyperparams_grid:
        print(f"\n=== Starting training with: {params} ===")
        
        model_name = params.get('model', 'yolo11s.pt')
        model = YOLO(model_name)
        
        results = model.train(
            data=data_path,
            epochs=params['epochs'],
            imgsz=params['imgsz'],
            batch=params['batch'],
            optimizer=params['optimizer'],
            lr0=params['lr0'],
            device=device,
            project='obstacle_tuning_high_res',
            name=f"{model_name[:-3]}_{params['optimizer']}_lr{params['lr0']}_b{params['batch']}_sz{params['imgsz']}",
            patience=50,
            cos_lr=True,          
            warmup_epochs=3,
            close_mosaic=10,    
            verbose=False,
            workers=2 
        )
        
        metrics = model.val(split='val')
        current_map = metrics.box.map  
        
        print(f"Validation mAP@0.5:0.95: {current_map:.4f}")
        print(f"Mean Precision: {metrics.box.mp:.4f} | Mean Recall: {metrics.box.mr:.4f}")
        source_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        current_model_name = f"model_{model_name[:-3]}_{params['optimizer']}_lr{params['lr0']}_b{params['batch']}.pt"
        if os.path.exists(source_path):
            shutil.copy(source_path, current_model_name)
            print(f"Current run model saved to: {current_model_name}")

        # Save the last model from the current run for comparison
        last_source_path = os.path.join(results.save_dir, 'weights', 'last.pt') # Save the last model from the current run for comparison
        if os.path.exists(last_source_path):
            shutil.copy(last_source_path, 'last.pt')
            print("Last run model saved to: last.pt")

        if current_map > best_map:
            best_map = current_map
            best_params = params
            # Save the best weights
            new_path = 'best_obstacle_model.pt'
            if os.path.exists(source_path):
                shutil.copy(source_path, new_path)
                best_model_path = new_path

        # Clear memory between iterations to prevent OOM
        del model, results, metrics
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"Best validation mAP: {best_map:.4f}")
    print(f"Best parameters: {best_params}")
    print(f"Best model saved as: {best_model_path}")

    # Final evaluation on the test set
    if best_model_path and os.path.exists(best_model_path):
        final_model = YOLO(best_model_path)
        test_metrics = final_model.val(data=data_path, split='test')
        print(f"\nFINAL TEST SET mAP@0.5:0.95: {test_metrics.box.map:.4f}")
        print("This is your model's expected real-world performance.")