# Lab 2 
#Part 1: MNIST Classification

We used PyTorch to classify handwritten digits with a CNN, Faster R-CNN, VGG16 and AlexNet. We trained the models locally in Jupyter on an NVIDIA GeForce RTX 3060 Laptop GPU.

## Dataset and preprocessing

We used MNIST with 55,000 training images, 5,000 validation images and the official 10,000 test images. We reused the same training and validation indices for all four models. We selected checkpoints using validation results before evaluating them on the test set.

For the CNN, we kept the grayscale images at 28 × 28 pixels and normalized them with mean 0.1307 and standard deviation 0.3081. We copied the supplied IDX files into the folder expected by torchvision when needed.

![MNIST samples](figures/cnn_samples.png)

## 1. CNN classifier

We built a CNN with two convolution layers containing 32 and 64 filters. Both used a 3 × 3 kernel, stride 1 and padding 1, followed by ReLU and 2 × 2 max pooling. We flattened the features and used fully connected layers with 128 and 10 outputs. Dropout2d of 0.25 and dropout of 0.5 reduced overfitting.

We trained for 10 epochs with Adam, a learning rate of 0.001, weight decay of 0.0001 and a batch size of 64. We used cross-entropy loss with raw logits and saved the checkpoint with the lowest validation loss.

The CNN reached 99.29% test accuracy, with a test loss of 0.021814. Training took 41.544190 seconds, excluding validation.

![CNN learning curves](figures/cnn_learning_curves.png)

![CNN confusion matrix](figures/cnn_confusion_matrix.png)

## 2. Faster R-CNN

MNIST provides digit labels without bounding boxes. We created one box per image from the pixels whose intensity was greater than 0.1. We repeated the grayscale channel three times and shifted the digit labels to 1-10 because label 0 is reserved for background.

We built Faster R-CNN from scratch with a small backbone containing two convolution layers with 32 and 64 channels, ReLU and max pooling. We used anchor sizes of 8, 16 and 24 pixels, aspect ratios of 0.5, 1 and 2, and RoIAlign with a 7 × 7 output. We kept the image size at 28 × 28 pixels.

We trained for 10 epochs with Adam, a learning rate of 0.001 and a batch size of 4. We selected the checkpoint with the highest validation accuracy. Epoch 9 reached 98.92% validation accuracy. For classification, we kept the highest-scoring detection for each image and counted missing detections as errors.

The saved model reached 98.90% test accuracy. Training took 2644.909476 seconds. The final epoch's training loss was 0.0287; this combines classification and bounding-box losses and is not a test cross-entropy loss.

![Faster R-CNN learning curves](figures/rcnn_learning_curves.png)

![Faster R-CNN confusion matrix](figures/rcnn_confusion_matrix.png)

## 3. CNN and Faster R-CNN comparison

The CNN achieved higher accuracy and F1 scores while training about 64 times faster. Faster R-CNN also learned to locate each digit, which added work beyond the classification task. For these single-digit images, the CNN was the more efficient choice.

We kept their losses separate because the models optimize different objectives. The CNN test loss measures classification error, while Faster R-CNN's recorded training loss also includes detection losses.

## 4. Fine-tuning VGG16 and AlexNet

We loaded ImageNet-pretrained weights and resized MNIST images directly to 224 × 224 pixels with three channels. We used ImageNet normalization and kept the original data split.

For VGG16, we fine-tuned the last convolution block. For AlexNet, we fine-tuned the last convolution layer. We replaced each final classifier layer with a layer containing 10 outputs and froze the remaining parameters. Both used Adam with a learning rate of 0.0001, weight decay of 0.0001, a batch size of 16 and three completed epochs.

We selected both checkpoints using the lowest validation loss, reached at epoch 2. VGG16 achieved 99.43% test accuracy with a test loss of 0.017559. AlexNet achieved 99.03% with a test loss of 0.025695.

![VGG16 learning curves](figures/vgg16_learning_curves.png)

![VGG16 confusion matrix](figures/vgg16_confusion_matrix.png)

![AlexNet learning curves](figures/alexnet_learning_curves.png)

![AlexNet confusion matrix](figures/alexnet_confusion_matrix.png)

### Results

| Model | Test accuracy | Macro F1 | Weighted F1 | Training time (s) | Completed epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 99.29% | 0.992847 | 0.992896 | 41.544190 | 10 |
| Faster R-CNN | 98.90% | 0.988977 | 0.988982 | 2644.909476 | 10 |
| VGG16 | 99.43% | 0.994246 | 0.994300 | 880.246967 | 3 |
| AlexNet | 99.03% | 0.990239 | 0.990298 | 212.156447 | 3 |

Training times include the training passes and exclude validation. We used different batch sizes and epoch counts, and VGG16 and AlexNet started with pretrained weights, so these timings describe our experiments rather than an equal-budget benchmark.

### Conclusion

VGG16 gave the highest test accuracy, but its improvement over the CNN was only 0.14 percentage points and training took about 21 times longer. AlexNet and Faster R-CNN were slower than the CNN and obtained lower accuracy in our runs.

We found the CNN to be the best balance between accuracy and training time for this task. We also learned how to adapt classification data for detection, fine-tune pretrained layers, and use validation results to choose a checkpoint without selecting it from test performance.
