#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('pip', 'install -q gradio')
#get_ipython().run_line_magic('pip', 'install fastbook')
#get_ipython().run_line_magic('pip', 'install -Uqq fastai')


# In[2]:


import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
from fastai.vision.all import *
import gradio as gr


# In[3]:


#!pip uninstall Pillow
#!pip install Pillow
# import PIL.Image
# if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
#     PIL.Image.Resampling = PIL.Image
# Now PIL.Image.Resampling.BICUBIC is always recognized.


# In[5]:


#import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath


# In[8]:


learn_inf = load_learner('NovaOrToastModel.pkl')
learn_inf.dls.vocab # Reminds us of the categories
categories = learn_inf.dls.vocab

def classify_image(img):
    pred, idx, probs = learn_inf.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192)) 
label = gr.outputs.Label()
examples = ["ToastTest.jpeg"]

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)

