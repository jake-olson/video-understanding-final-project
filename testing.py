import torch
from timesformer.models.vit import TimeSformer

model_file = '~/Desktop/cs89-final/TimeSformer/configs/Kinetics/TimeSformer_divST_8x32_224_.pyth'
model_divSpaceTime = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model=(model_file))

dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)

#Dummy video in order to prove that it is working properly
pred = model_divSpaceTime(dummy_video,) # (2, 400)

# Instantiating several different versions of TimeSformer
model_file_joint = '~/Desktop/cs89-final/TimeSformer/configs/Kinetics/TimeSformer_jointST_8x32_224_.pyth'
model_file_spaceOnly = '~/Desktop/cs89-final/TimeSformer/configs/Kinetics/TimeSformer_spaceOnly_8x32_224_.pyth'
model_joint = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='joint_space_time',  pretrained_model=(model_file_joint))
model_space_only = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='space_only',  pretrained_model=(model_file_spaceOnly))

# Evaluate the results on the different datasets
howto100_file = '/Users/davidmatusz/Desktop/cs89_final/TimeSformer/train.csv'
kinetics600_file = '/Users/davidmatusz/Desktop/cs89_final/TimeSformer/kinetics-dataset/k600_extractor.sh'
# Limited due to constraints
numOfClasses_600 = 10
numOfClasseshowTo100 = 20
