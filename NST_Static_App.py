# -*- coding: utf-8 -*-

# =============================================================================
# Importing Necessary Libraries
# =============================================================================
from PIL import Image
import streamlit as st
import os
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False
import time
import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_dir = BASE_DIR + 'images/'


"""## Neural Style Transfer
In this post, I will be introducing a tool I developed based on an algorithm [**'Neural Style Transfer'**](https://arxiv.org/abs/1508.06576)!
Feel free to apply it to your data.

**N**eural **S**tyle **T**ransfer is a technique to extract features i.e.the appearance or visual style, from the 
reference image also known as **'Style Image'** and to apply it to your image. NST algorithms have been implemented via deep 
neural networks. A common application of **'NST'** is a creation of artificial work from photographs/paintings by transferring
the features of the paintings to the user-supplied photographs i.e, the **'Content Image'**. 
Here, is an interesting [**article**](https://www.christies.com/features/A-collaboration-between-two-artists-one-human-one-a-machine-9332-1.aspx)
stating the application of the same.

Below is an example that was generated using NST. Here, the content image is 
taken digitally somewhere in Austria and the style image ([**Starry Night**](https://www.vincentvangogh.org/starry-night.jsp))
 is from famous artist **Van Gogh**. Now imagine what happens if Van Gogh tried to paint the content image, do you
know how would it look like? The below image shows it would look something like this.  

"""
st.image(Image.open(img_dir+"linkedin.jpg"), width=512)


def returnNotMatches(a, b):
    return [x for x in a if x not in b]

# datetime object containing current date and time
now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# =============================================================================
# Upload User Style/Content image
# =============================================================================

uploaded_file = st.file_uploader("Choose a Content Image file", type=["jpg","jpeg","png", "JPEG"])

if uploaded_file is not None:
    data = Image.open(uploaded_file)
    data = data.convert("RGB")
    data.save( img_dir + "ContentImages/"+'Your-Content-Image.jpg')
    # st.image(data, use_column_width=True)

uploaded_file1 = st.file_uploader("Choose a Style Image file", type=["jpg","jpeg","png", "JPEG"])
if uploaded_file1 is not None:
    data1 = Image.open(uploaded_file1)
    data1 = data1.convert("RGB")
    data1.save( img_dir + "styleImages/"+'Your-Style-Image.jpg')
    # st.image(data, use_column_width=True)
    st.empty()

# =============================================================================
# Dropdown list of Style and Content Image
# =============================================================================

contentImageNames = os.listdir(img_dir+'contentImages/')

# @st.cache()
contentImageNames111 = ["House-in-Austria.jpg", "Walhalla-Regensburg.jpg", "River.JPG", "King-of-Walhalla.JPG",
                        "River-Bridge-Path.JPG", "River-Bridge-Tree.JPG", "Riverfront.JPG", "Riverfront-Cycle.JPG",
                        "Tree-Springs.JPG", "Tree-Springs1.JPG", "View-From-Walhalla.JPG"]
aa = returnNotMatches(contentImageNames, contentImageNames111)
# aa = list(set(contentImageNames).difference(set(contentImageNames111)))

contentImage = st.selectbox("Please Select a Content Picture", contentImageNames, len(contentImageNames)-1)
selectedContentImage = img_dir + "contentImages/" + contentImage
# st.image(Image.open(selectedContentImage), caption=contentImage, use_column_width=True)


styleImageNames = os.listdir(img_dir+'styleImages/')
styleImageNames111 = ["La-Mousme.jpg", "Self-Potrait.jpg", "Starry-Night.jpg", "Tuebingen-Neckarfront.jpg",
      "Vassily-Kandinsky.jpg", "Waves.jpg", "Starry-Night-Over-the-Rhone.jpg","Style-Art-Image.jpg"]
bb = list(set(styleImageNames).difference(set(styleImageNames111)))


styleImage = st.selectbox("Please Select a Style Picture", styleImageNames, len(styleImageNames)-1)
selectedStyleImage = img_dir + "styleImages/"+ styleImage
# st.image(Image.open(selectedStyleImage), caption=styleImage, use_column_width=True)

# Set up some global values here
content_path = selectedContentImage
style_path = selectedStyleImage

"""These are input content and style images. We hope to **"create"** an image with the content
 of our content image, but with the style of the style image."""


# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


"""### Intuitive Explanation of the NST Technique
As part of the implementation of NST I am using [VGG16](https://keras.io/applications/#vgg16) pretrained network for our custom images. 
And VGG networks are trained on an image with each channel normalized by `mean = [103.939, 116.779, 123.68]`
Hence, we also need to normalize our input images as per the VGG16 model standard.

The below figure shows `model.summary()` of VGG16 model:
"""
image = Image.open(img_dir + '/vgg16.png')
st.image(image, caption='VGG16 Model Summary',use_column_width=True)

"""
As we go deeper into the layers the features extracted from the image, increases.
The last layer of VGG16 model has the highest number of features i.e. shapes. And the shallow layers
contains a small number of extracted features which is useful to apply on our style image to extract 
vital pieces of information like the stroke of the brush have happened into the style image to adopt.
Hence, our Content image is only trained on the last layer of VGG16 and our Style image is trained on 
all the layers except the last layer of VGG16.

Here, for your image you can choose which layer to consider for Content image and Style image.
You can try different combinations and can compare the results later on. As well as one can use 
this tool for hyperparameter tuning.
"""



ln = ['input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1','block2_conv2','block2_pool','block3_conv1','block3_conv2','block3_conv3',
 'block3_pool','block4_conv1','block4_conv2','block4_conv3','block4_pool','block5_conv1','block5_conv2','block5_conv3']
style_layers = st.multiselect('Please select layer/s to train Style image on', ln[1:], default=style_layers)
content_layers = st.multiselect('Please select layer/s to train Content image on', ln[1:], default=content_layers)



num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

st.sidebar.image(Image.open(selectedContentImage), caption="Content Image", width=200)

st.sidebar.image(Image.open(selectedStyleImage), caption="Style Image", width=200)

"""
In NST the loss function is the summation of loss between input image with the content image (content loss)
 and loss between an input image and Style image (style loss).
And also different weightage for content loss and style loss are given to optimize respective losses more.
"""

# =============================================================================
# Hyperparameter Selection
# =============================================================================
content_weight = st.number_input('Please Insert Content Weight', value=1e3)
st.write('Selected Content Weight is: {}'.format(content_weight))
style_weight = st.number_input('Please Insert Style Weight', value=1e-2)
st.write('Selected Style Weight is: {}'.format(style_weight))
num_iterations = st.slider('Number of iterations for training?', 0, 5000, 100)

row_col = st.number_input('Please Select Number of Images you want to plot', value=10)


## Training

if st.button('Start Training'):
    st.text("Training Started")

# =============================================================================
# Data Deletion and Celebration
# =============================================================================
    
if len(aa) != 0 or len(bb) != 0:
    vv = st.button("Delete Data")
    if vv:
        if len(bb) != 0:
            os.remove(img_dir + "styleImages/"+ "Your-Style-Image.jpg")
        if len(aa) != 0:
            os.remove(img_dir + "contentImages/" + "Your-Content-Image.jpg")

st.markdown("## Party time!")
st.write("Yay! You're done with this Training and Generation of NST image. Click below to celebrate.")
btn = st.button("Celebrate!")
if btn:
    st.balloons()
    
'''
#### Please feel free to write me any suggestions you have
You can reach me via [**LinkedIn**](https://www.linkedin.com/in/vedant-parikh-04923a41/) or [**E-mail**](mailto:vedant.parikh@outlook.com)
and feel free to have look at other Machine Learning / Data Science projects on my [**Github**](https://github.com/vedantparikh) page.
'''
