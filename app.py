import cv2, PIL, glob, os   
import numpy as np 
import matplotlib.pyplot as plt 

from flask import Flask, render_template, request
#from keras.models import load_model 
#from tensorflow.keras.utils import normalize 

from diffusers import StableDiffusionInpaintPipeline
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F 
from torchvision.utils import save_image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
import torchvision.transforms as T 

#import init 
#from init import * 
# import gc

# gc.collect() 

# torch.cuda.empty_cache()

app = Flask(__name__)

# Accessing GPU 
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()


# define validation transform
val_transforms = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2()
])

#model_initialization = init.UNet
class BaseClass(nn.Module):
    def training_step(self, batch):
        inputs, targets = batch        
        preds = self(inputs)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(preds, targets)
        return loss
    
    def validation_step(self, batch, score_fn):
        inputs, targets = batch
        preds = self(inputs)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(preds, targets)
        score = score_fn(preds, targets)
        return {'val_loss': loss.detach(), 'val_score': score}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = sum(batch_losses)/len(batch_losses)
        
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = sum(batch_scores)/len(batch_scores)
        
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score}
    
    def epoch_end(self, epoch, nEpochs, results):
        print("Epoch: [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score:{:.4f}".format(
                        epoch+1, nEpochs, results['train_loss'], results['val_loss'], results['val_score']))

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size = 3,stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)

def copy_and_crop(down_layer, up_layer):
    b, ch, h, w = up_layer.shape
    crop = T.CenterCrop((h, w))(down_layer)
    return crop
    
class UNet(BaseClass):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList([
            conv_block(in_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256), 
            conv_block(256, 512)
        ])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottle_neck = conv_block(512, 1024)
        
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        
        self.decoder = nn.ModuleList([
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64)
        ])
        
        self.final_layer = nn.Conv2d(64, out_channels, 1, 1)
        
    def forward(self, x):
        skip_connections = []
        
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottle_neck(x)
        
        for ind, layer in enumerate(self.decoder):
            x = self.up_samples[ind](x)
            y = copy_and_crop(skip_connections.pop(), x)
            x = layer(torch.cat([y, x], dim=1))
        
        x = self.final_layer(x)
        
        return x 


# pytorch Model loading 
device = torch.device('cuda')  
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

model = to_device(UNet(3, 1), device) 

loaded_model_pretrained = model 
loaded_model_pretrained.load_state_dict(torch.load("./models/face_tune_unet_segmentation_state_dict.pt")) 
loaded_model_pretrained.to(device) 
loaded_model_pretrained.eval() 


# Function for creating mask image 
def mask_image_creation(img_path, loaded_model): 

	img = np.array(PIL.Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
	img_t = val_transforms(image=img)['image'].to(device)

	logits = loaded_model(img_t.unsqueeze(0)).detach().cpu()
	preds = F.sigmoid(logits)
	preds = (preds>0.5).float().detach().cpu()
	preds = preds[0].permute(1,2,0)
	#plt.imshow(preds, cmap='gray')
	#plt.axis('off')

	# saving the mask image 
	#print(preds.shape) 
	new_shape = preds.permute(2,0,1)
	#print("after shape change: ", (new_shape.shape)) 
	#save_image(new_shape, "./predicted_mask_img/mask_93.png") 
	return new_shape 

# Function for processing image 
def preprocessed_image(url_path):
    img_prepro = PIL.Image.open(url_path).convert("RGB").resize((256, 256))
    return img_prepro

# Function for inpainting 
def wrinkle_remove_using_stable_diffusion_2_inpaint(img_url, mask_url, streng, num_inferen):
	
	img = preprocessed_image(img_url)
	mask = preprocessed_image(mask_url)
 

	pipe = StableDiffusionInpaintPipeline.from_pretrained(
		"stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
	)

	#device = "cuda" if torch.cuda.is_available() else "cpu"
	pipe = pipe.to("cuda")   

	prompt = ""
	inpaint = pipe(prompt=prompt, image=img, mask_image=mask, strength=streng, num_inference_steps=num_inferen).images[0]
	return inpaint


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/images")
def about_page():
	return render_template('image.html') 
    #return "this is the string text"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		all_path = ["./static/original_img", "./static/mask_img", "./static/inpainted_img"]
		for fol in all_path:
			all_file = os.listdir(fol) 
			for file_ in all_file:
				os.remove(os.path.join(fol, file_))  


		img_path = "./static/original_img/original.jpg" #+ img.filename	

		img.save(img_path)

		#p = predict_label(img_path) 
		predicted_mask_img = mask_image_creation(img_path, loaded_model_pretrained) 
		# generating name of the mask image 
		mask_path = "./static/mask_img/mask.jpg"  #+ img.filename 
		# saving the mask image 
		#plt.imsave(mask_path, predicted_mask_img) 
		save_image(predicted_mask_img, mask_path) 

		# inpainting 
		strength = float(request.form['strength']) 
		num_inference_steps = int(request.form['num_inference_steps'])  

	inpainted_image = wrinkle_remove_using_stable_diffusion_2_inpaint(img_path, mask_path, strength, num_inference_steps) 
	inpainted_path = "./static/inpainted_img/inpainted.jpg"  #+ img.filename 
	inpainted_image.save(inpainted_path) 


	return render_template("index.html", prediction = "predicted", strength=strength, inference = num_inference_steps)  


if __name__ =='__main__':
	#app.debug = True
	app.run()
    