# Speech-based stress detection using a cross-modal transfer learning approach
```
pipenv shell
python train_stage.py --dir RGB/oversampling/ --until_layer 14 --n_epochs 500 --batch_size 32 --use_gpu --do_test
```