LOSS_GAMMA = 10  # from authors, not optimized
LOSS_NEED_INTERMEDIATE_LAYERS = True
# after_conv for features from the last convolution layer,
# before_class for features before classifier
# before_bottleneck for features before bottleneck
FEATURES = "after_conv"
UNK_VALUE = -100  # torch default
IS_UNSUPERVISED = True

GRADIENT_REVERSAL_LAYER_ALPHA = 1.0
FREEZE_BACKBONE_FEATURES = True
# possible for AlexNet: 0, 2, 4, 6, 8, 10
# possible for ResNet: 0, 1, 3, 33, 72, 129, 141, 159
FREEZE_LEVEL = 141
BATCH_SIZE = 64

NUM_WORKERS = 4

N_EPOCHS = 401
STEPS_PER_EPOCH = 10
VAL_FREQ = 1
SAVE_MODEL_FREQ = 400

################### Model dependent parameters #########################

CLASSES_CNT = 31
# MODEL_BACKBONE = "alexnet" # alexnet resnet50 resnet50_rich vanilla_dann
MODEL_BACKBONE = "resnet50_rich"
NEED_ADAPTATION_BLOCK = False
BOTTLENECK_SIZE = 256
DOMAIN_HEAD = "vanilla_dann"
DOMAIN_LOSS = 0.5
CLASSIFICATION_LOSS = 2 - DOMAIN_LOSS
BACKBONE_PRETRAINED = True
IMAGE_SIZE = 224
DATASET = "office-31"
# SOURCE_DOMAIN = "amazon"
# TARGET_DOMAIN = "webcam"

SOURCE_DOMAIN = "webcam"
TARGET_DOMAIN = "amazon"

# CLASSES_CNT = 10
# MODEL_BACKBONE = "mnist_dann"
# DOMAIN_HEAD = "mnist_dann"
# BACKBONE_PRETRAINED = False
# NEED_ADAPTATION_BLOCK = False
# BLOCKS_WITH_SMALLER_LR = 0
# IMAGE_SIZE = 28
# DATASET = "mnist"
# SOURCE_DOMAIN = "mnist"
# TARGET_DOMAIN = "mnist-m"

# assert (MODEL_BACKBONE == "alexnet" or
#         (MODEL_BACKBONE != "alexnet" and not NEED_ADAPTATION_BLOCK and BLOCKS_WITH_SMALLER_LR == 0)), \
#        "can't use adaptation block with non-alexnet"
