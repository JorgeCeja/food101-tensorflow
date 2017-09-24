# Food-101 Dataset Using Transfer Learning

Inspired by HBO’s Silicon Valley “Not Hotdog” App, I set out to classify not only hotdogs but 101 categories of different foods. The The other goal was to use data augmentation and transfer learning and data augmentation to achive fast(er) training time and accuracy. 

### Prerequisites

- [Food-101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
- Nvidia GPU or cloud GPU instances for training 
- Tensoflow 
- Keras
- Numpy

## Getting Started

1. `git clone ` + repo URL
2. cd to repo
3. `pip install -r /requirements/requirement.txt` if packages are not yet installed
- Train model: `python food_101.py -m train `
- Test model: `python food_101.py -m test -i test_image.jpg`

## History

1. Initial test with 48% accuracy after 2 epochs!
2. Add command line arguments including dropout

## Built With

* [Tensoflow](https://www.tensorflow.org) - Software library for numerical computation using data flow graphs
* [Keras](https://keras.io) - Deep Learning library
* [Matplotlib](https://matplotlib.org) - Python 2D plotting library
* [Numpy](http://www.numpy.org) - Package for scientific computing

## Contributing

1. Fork it! Star it?
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Authors

* **Jorge Ceja** - *Initial work* - [Account](https://github.com/JorgeCeja)

## Acknowledgments

* Food-101 – Mining Discriminative Components with Random Forests - [Research Paper](https://www.vision.ee.ethz.ch/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf)
* Deep Residual Learning for Image Recognition - [arXiv](https://arxiv.org/abs/1512.03385)
* Going Deeper with Convolutions ("Inception") - [arXiv](https://arxiv.org/abs/1409.4842)