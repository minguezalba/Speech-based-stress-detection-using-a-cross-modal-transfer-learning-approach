# Speech-based stress detection using a cross-modal transfer learning approach

Run it on CPU:
```python 2_image_model_training.py --dir RGB/balanced/ --until_layer 2 --n_epochs 100 --batch_size 100```

Run it on GPU:
```python 2_image_model_training.py --dir RGB/balanced/ --until_layer 2 --n_epochs 100 --batch_size 100 --use_gpu```

Run it and save output on file:
```python 2_image_model_training.py --dir RGB/balanced/ --until_layer 2 --n_epochs 2 --batch_size 100 --use_gpu 2>&1 | tee logs/exp.txt```