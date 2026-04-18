```python
def training(train_data, train_label, save_path):
    pass

def iterative_SSL(train_data, train_label, unlabel_data, threashold):
    model = training(train_data, train_label)
    for _ in 1 to n_iter:
        filtered_SSL_data, pseudo_label = gen_pseudo_label(model, unlabel_data)
        model, train(train_data + filtered_SSL_data, train_label + pseudo_label)
```