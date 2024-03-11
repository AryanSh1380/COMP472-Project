# Aryan Shirazi - 40119594
# Adnan Saab - 40075504
# Karim Tabbara - 40157871

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

# read data from csv file
data = pd.read_csv("Labeled Data.csv")

# get the count of each of the labels
class_distribution_count = data['label'].value_counts()

# 1) CLASS DISTRIBUTION : plot bar graph using matplotlib

plt.bar(class_distribution_count.index, class_distribution_count.values)
plt.xlabel("Classes")
plt.ylabel("Count")
plt.title("Class Distribution")


# 2) PLOT RANDOM SAMPLE IMAGES WITH THEIR PIXEL INTENSITY DISTRIBUTION

dataFrame = pd.DataFrame(data)

# Sample images from each set based on the labels
# And create the 5x5 subplot grid to display them
# Also create 5x5 subplot grid for the pixel intensity of each sampled picture
NeutralSample = dataFrame.loc[dataFrame['label'] == 'Neutral'].sample(n=25)
neutral_fig , (neutral_axes1, neutral_axes2, neutral_axes3, neutral_axes4, neutral_axes5) \
    = plt.subplots(5, 5, figsize=(15, 15), tight_layout=True)
neutral_pixel, (neutral_pix_ax1, neutral_pix_ax2, neutral_pix_ax3, neutral_pix_ax4, neutral_pix_ax5) \
    = plt.subplots(5, 5, figsize=(15,15), tight_layout=True)

FocusedSample = dataFrame.loc[dataFrame['label'] == 'Focused'].sample(n=25)
focused_fig , (focused_axes1, focused_axes2, focused_axes3, focused_axes4, focused_axes5) \
    = plt.subplots(5, 5, figsize=(15, 15), tight_layout=True)
focused_pixel, (focused_pix_ax1, focused_pix_ax2, focused_pix_ax3, focused_pix_ax4, focused_pix_ax5) \
    = plt.subplots(5, 5, figsize=(15,15), tight_layout=True)

SurprisedSample = dataFrame.loc[dataFrame['label'] == 'Surprised'].sample(n=25)
surprised_fig , (surprised_axes1, surprised_axes2, surprised_axes3, surprised_axes4, surprised_axes5) \
    = plt.subplots(5, 5, figsize=(15, 15), tight_layout=True)
surprised_pixel, (surprised_pix_ax1, surprised_pix_ax2, surprised_pix_ax3, surprised_pix_ax4, surprised_pix_ax5) \
    = plt.subplots(5, 5, figsize=(15,15), tight_layout=True)

HappySample = dataFrame.loc[dataFrame['label'] == 'Happy'].sample(n=25)
happy_fig , (happy_axes1, happy_axes2, happy_axes3, happy_axes4, happy_axes5) \
    = plt.subplots(5,5,figsize=(15, 15), tight_layout=True)
happy_pixel, (happy_pix_ax1, happy_pix_ax2, happy_pix_ax3, happy_pix_ax4, happy_pix_ax5) \
    = plt.subplots(5, 5, figsize=(15,15), tight_layout=True)




# Split Neutral sample into 5 groups (each group represents 1 row in the 5x5 grid)
NeutralSample_Set1 = NeutralSample[0:5]
NeutralSample_Set2 = NeutralSample[5:10]
NeutralSample_Set3 = NeutralSample[10:15]
NeutralSample_Set4 = NeutralSample[15:20]
NeutralSample_Set5 = NeutralSample[20:25]
# Display the 25 Neutral images by populating each row in the grid
# And display each image's pixel intensity distribution
for i, (index, row) in enumerate(NeutralSample_Set1.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    neutral_axes1[i].imshow(image)
    neutral_axes1[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    neutral_pix_ax1[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    neutral_pix_ax1[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    neutral_pix_ax1[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    neutral_pix_ax1[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(NeutralSample_Set2.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    neutral_axes2[i].imshow(image)
    neutral_axes2[i].axis('off')
    neutral_pix_ax2[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    neutral_pix_ax2[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    neutral_pix_ax2[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    neutral_pix_ax2[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(NeutralSample_Set3.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    neutral_axes3[i].imshow(image)
    neutral_axes3[i].axis('off')
    neutral_pix_ax3[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    neutral_pix_ax3[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    neutral_pix_ax3[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    neutral_pix_ax3[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(NeutralSample_Set4.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    neutral_axes4[i].imshow(image)
    neutral_axes4[i].axis('off')
    neutral_pix_ax4[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    neutral_pix_ax4[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    neutral_pix_ax4[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    neutral_pix_ax4[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(NeutralSample_Set5.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    neutral_axes5[i].imshow(image)
    neutral_axes5[i].axis('off')
    neutral_pix_ax5[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    neutral_pix_ax5[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    neutral_pix_ax5[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    neutral_pix_ax5[i].set_title("Pixel Intensity")




# Split Focused sample into 5 groups (each group represents 1 row in the 5x5 grid)
FocusedSample_Set1 = FocusedSample[0:5]
FocusedSample_Set2 = FocusedSample[5:10]
FocusedSample_Set3 = FocusedSample[10:15]
FocusedSample_Set4 = FocusedSample[15:20]
FocusedSample_Set5 = FocusedSample[20:25]
# Display the 25 Focused images by populating each row in the grid
# And display each image's pixel intensity distribution
for i, (index, row) in enumerate(FocusedSample_Set1.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    focused_axes1[i].imshow(image)
    focused_axes1[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    focused_pix_ax1[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    focused_pix_ax1[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    focused_pix_ax1[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    focused_pix_ax1[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(FocusedSample_Set2.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    focused_axes2[i].imshow(image)
    focused_axes2[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    focused_pix_ax2[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    focused_pix_ax2[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    focused_pix_ax2[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    focused_pix_ax2[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(FocusedSample_Set3.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    focused_axes3[i].imshow(image)
    focused_axes3[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    focused_pix_ax3[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    focused_pix_ax3[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    focused_pix_ax3[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    focused_pix_ax3[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(FocusedSample_Set4.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    focused_axes4[i].imshow(image)
    focused_axes4[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    focused_pix_ax4[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    focused_pix_ax4[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    focused_pix_ax4[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    focused_pix_ax4[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(FocusedSample_Set5.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    focused_axes5[i].imshow(image)
    focused_axes5[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    focused_pix_ax5[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    focused_pix_ax5[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    focused_pix_ax5[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    focused_pix_ax5[i].set_title("Pixel Intensity")




# Split Surprised sample into 5 groups (each group represents 1 row in the 5x5 grid)
SurprisedSample_Set1 = SurprisedSample[0:5]
SurprisedSample_Set2 = SurprisedSample[5:10]
SurprisedSample_Set3 = SurprisedSample[10:15]
SurprisedSample_Set4 = SurprisedSample[15:20]
SurprisedSample_Set5 = SurprisedSample[20:25]
# Display the 25 Surprised images by populating each row in the grid
# And display each image's pixel intensity distribution
for i, (index, row) in enumerate(SurprisedSample_Set1.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    surprised_axes1[i].imshow(image)
    surprised_axes1[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    surprised_pix_ax1[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    surprised_pix_ax1[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    surprised_pix_ax1[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    surprised_pix_ax1[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(SurprisedSample_Set2.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    surprised_axes2[i].imshow(image)
    surprised_axes2[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    surprised_pix_ax2[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    surprised_pix_ax2[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    surprised_pix_ax2[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    surprised_pix_ax2[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(SurprisedSample_Set3.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    surprised_axes3[i].imshow(image)
    surprised_axes3[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    surprised_pix_ax3[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    surprised_pix_ax3[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    surprised_pix_ax3[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    surprised_pix_ax3[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(SurprisedSample_Set4.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    surprised_axes4[i].imshow(image)
    surprised_axes4[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    surprised_pix_ax4[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    surprised_pix_ax4[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    surprised_pix_ax4[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    surprised_pix_ax4[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(SurprisedSample_Set5.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    surprised_axes5[i].imshow(image)
    surprised_axes5[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    surprised_pix_ax5[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    surprised_pix_ax5[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    surprised_pix_ax5[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    surprised_pix_ax5[i].set_title("Pixel Intensity")




# Split Happy sample into 5 groups (each group represents 1 row in the 5x5 grid)
HappySample_Set1 = HappySample[0:5]
HappySample_Set2 = HappySample[5:10]
HappySample_Set3 = HappySample[10:15]
HappySample_Set4 = HappySample[15:20]
HappySample_Set5 = HappySample[20:25]
# Display the 25 Happy images by populating each row in the grid
# And display each image's pixel intensity distribution
for i, (index, row) in enumerate(HappySample_Set1.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    happy_axes1[i].imshow(image)
    happy_axes1[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    happy_pix_ax1[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    happy_pix_ax1[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    happy_pix_ax1[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    happy_pix_ax1[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(HappySample_Set2.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    happy_axes2[i].imshow(image)
    happy_axes2[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    happy_pix_ax2[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    happy_pix_ax2[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    happy_pix_ax2[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    happy_pix_ax2[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(HappySample_Set3.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    happy_axes3[i].imshow(image)
    happy_axes3[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    happy_pix_ax3[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    happy_pix_ax3[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    happy_pix_ax3[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    happy_pix_ax3[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(HappySample_Set4.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    happy_axes4[i].imshow(image)
    happy_axes4[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    happy_pix_ax4[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    happy_pix_ax4[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    happy_pix_ax4[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    happy_pix_ax4[i].set_title("Pixel Intensity")
for i, (index, row) in enumerate(HappySample_Set5.iterrows()):
    image_path = row['path']
    image = imread(image_path)
    happy_axes5[i].imshow(image)
    happy_axes5[i].axis('off')
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    happy_pix_ax5[i].hist(red.flatten(), bins=256, density=False, color='red', alpha=0.5)
    happy_pix_ax5[i].hist(green.flatten(), bins=256, density=False, color='green', alpha=0.4)
    happy_pix_ax5[i].hist(blue.flatten(), bins=256, density=False, color='blue', alpha=0.3)
    happy_pix_ax5[i].set_title("Pixel Intensity")

plt.show()