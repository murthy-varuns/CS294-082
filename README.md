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

## Versioning

We use [PyPI](https://pypi.org/project/deepcompressor/) for versioning.

## Authors

* **Varun Murthy** - *murthy@berkeley.edu* 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Hat tip to Dr. Gerald Friedland *fractor@eecs.berkeley.edu* of the Department of EECS, University of California at Berkeley.
