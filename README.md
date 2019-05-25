# CS294-082
***Public Repository for CS 294-082 - Experimental Design for Machine Learning on Multimedia Data (Graduate) - Coursework (Spring '19)***

Hello! I'm glad you're here. Here is an overview of the project from the original report:

> "We use the VGG16Net model developed by the Visual Geometry Group at the University of Oxford and repeatedly train it on the MNIST          dataset while attempting to reduce its parameters without significantly reducing model accuracy."

### Prerequisites

Clone the repository and create a new virtual environment on Anaconda or virtualenv (I used Anaconda) to install all the needed dependencies.

`pip install -r requirements.txt`

Run `from deepcompressor import checker` and `checker.ensure_background()` to ensure that there are no problems with any dependencies. 

## Deep Compression 

### Step 1
Run `import deepcompressor` and create a new CapacityEstimator `ce = CapacityEstimator()`.
### Step 2
Create the model.
```
vgg16 = model('vgg16', True)
vgg16.set_capacity_estimator(ce)
vgg16.parallelize()
vgg16.prepare_dataset()
```
### Step 3
Set the training parameters.
```
num_epochs = 6
max_epochs_stop = 3
num_classes = 10
batch_size = 10
learning_rate = 0.001
print_every = 1
```
### Step 4
Create a DeepCompressor with a Model and its Trainer.
```
training_params = [num_epochs, max_epochs_stop, num_classes, batch_size, learning_rate, print_every]
deep_compressor = deep_compressor(vgg16, trainer(vgg16, training_params))
```
### Step 5
Repeatedly train, test and squeeze the network.
```
deep_compressor.train(dropout=100)
deep_compressor.test() # ce.estimate(25088, 4096, 4096, 256, 10)
deep_compressor.squeeze(100)
deep_compressor.train(dropout=100)
deep_compressor.test() # ce.estimate(25088, 4096-100, 4096, 256, 10)
deep_compressor.squeeze(100)
deep_compressor.train(dropout=100)
deep_compressor.test() # ce.estimate(25088, 4096-200, 4096, 256, 10)
deep_compressor.squeeze(100)
deep_compressor.train(dropout=100)
deep_compressor.test() # ce.estimate(25088, 4096-300, 4096, 256, 10)
deep_compressor.squeeze(100)
deep_compressor.train(dropout=100)
deep_compressor.test() # ce.estimate(25088, 4096-400, 4096, 256, 10)
deep_compressor.squeeze(100)
deep_compressor.train(dropout=100)
deep_compressor.test() # ce.estimate(25088, 4096-500, 4096, 256, 10)
```
### Step 6
Plot Generalization vs Capacity and save the figure if you want to.
```
df = pd.DataFrame(data=deep_compressor.model.history)
sns.set(style="darkgrid")
fig = sns.lineplot(x='cap', y='acc', data=df);
fig.set(ylabel='Generalization', xlabel='Memory Equivalent Capacity');
fig.get_figure().savefig('GC_curve.jpg')
```

## Versioning

We use [PyPI](https://pypi.org/project/deepcompressor/) for versioning.

## Authors

* **Varun Murthy** - *murthy@berkeley.edu* 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Hat tip to Dr. Gerald Friedland *fractor@eecs.berkeley.edu* of the Department of EECS, University of California at Berkeley.
