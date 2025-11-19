import numpy as np
from PIL import Image
import pygame
import random

pygame.init()
font=pygame.font.Font("SpaceMono-Bold.ttf",13)
small_font=pygame.font.Font("SpaceMono-Bold.ttf",12)

#allowing multiple images (input images, feature maps, predicted images) for better visualization

imgcountH=3
imgcountW=2

img_pad_x=60
img_pad_y=40

imgw=28
imgh=28

scale_fac=5

win_w=imgcountW*imgw*scale_fac+img_pad_x*(imgcountW+1)
win_h=imgcountH*imgh*scale_fac+img_pad_y*(imgcountH+1)

win=pygame.display.set_mode((win_w,win_h))
bg_colour=(22, 23, 26)
lighter_bg_colour=(40,42,52)
pygame.display.set_caption("CNN - recognizing digits")
icon = pygame.image.load(r"media\test_img.ico").convert_alpha()
pygame.display.set_icon(icon)
# convolutional neural net

class filter3x3: # the kernel that slides over the image to create a feature map
    def __init__(self):
        self.filter=np.array([
            [0.1,0.1,0.1],
            [0.1,0.1,0.1],
            [0.1,0.1,0.1]],dtype=np.float16) # numbers represent weights to multiply the pixel value by
        self.bias=0
       
   
         
class img: # for regular unprocessed images
    def __init__(self,filepath=r"media\test_img.png",index=1): # index represents the position in the window (top left to bottom right)
       
        self.filepath=filepath

        self.unprocessed_img=Image.open(filepath)

        self.img=np.array(self.unprocessed_img,dtype=np.float16)/255 if filepath is not None else None# normalized image values

        self.height,self.width=self.img.shape # image height, width

        self.index=index

        self.label="input image"

        self.tlx=0 #top left rect corner
        self.tly=0
        adj_index=self.index//2
        self.tly=img_pad_y*(adj_index+1)+imgh*(adj_index)*scale_fac # the top left coordinate calculation needs fixing as it only works with a imgcountH of 2

        if self.index % 2 ==0:
            self.tlx=img_pad_x
           
        else:
            self.tlx=img_pad_x*2+imgw*scale_fac
        self.text_colour=(222, 215, 202)
        self.text=font.render(self.label,True,self.text_colour)

        #print(self.img)
    def draw(self):

        pygame.draw.rect(win,(255,255,255),(self.tlx,self.tly,imgw*scale_fac,imgh*scale_fac))

        for y in range(self.height):
            for x in range(self.width):
                pix=self.img[y][x]*255
                pix_x=self.tlx+x*scale_fac #pixel top left corner co - ordinate
                pix_y=self.tly+y*scale_fac

                pygame.draw.rect(win,(pix,pix,pix),(pix_x,pix_y,scale_fac,scale_fac))

        win.blit(self.text,(self.tlx,self.tly-img_pad_y/1.5)) # add image label eg input image, feature map 1 etc
   
class conv_img: #img with added functions like passing a kernel over the image to create a feature map
    def __init__(self,image,fil,using_activation=True,maxpool=True):
        self.input_img=image
        self.filter=fil
        self.stride=1

        self.using_activation=using_activation #whether or not to use ReLU or similar activation. Likely false for visualizations

        self.filterw=3
        self.filterh=3

        self.padx=1
        self.pady=1
        # since this is created at init it will be the original image IMPORTANT if you want to change this you must use the og image
        self.padded= np.pad(self.input_img.img,((self.pady,self.pady),(self.padx,self.padx)),mode='constant',constant_values=0)

        self.width=int( (self.input_img.width + 2*self.padx-self.filterw)/self.stride + 1)
        self.height=int( (self.input_img.height + 2*self.pady-self.filterh)/self.stride + 1)

        self.output_img=np.array([[0 for _ in range(self.width)] for _ in range(self.height)],dtype=np.float16)

        self.label="max pooled" if maxpool else "feature map"
        self.label+="" if using_activation else " UNACTIVATED"

        self.text_colour=(222, 215, 202)
        self.text=font.render(self.label,True,self.text_colour)

        self.maxpool=maxpool
        self.maxpoolW=2
        self.maxpoolH=2
        self.maxpoolStride=2

        self.maxpool_img_w=int((self.width-self.maxpoolW)/self.maxpoolStride+1)
        self.maxpool_img_h=int((self.height-self.maxpoolH)/self.maxpoolStride+1)

        self.maxpool_img=np.zeros((self.maxpool_img_h,self.maxpool_img_w),dtype=np.float16)

        self.maxpool_img_indexes=np.zeros((self.maxpool_img_h,self.maxpool_img_w,2),dtype=np.uint16) # the 2 is for 2 cooridinates, this is done so that we can differentiate the values lates

    def produce_image(self):
        for h in range(self.height):
            for w in range(self.width):
                value=0
                for fh in range(self.filterh):
                    for fw in range(self.filterw):
                        value+=self.filter.filter[fh][fw]*self.padded[h*self.stride+fh][w*self.stride+fw]
                #print(value)
                if self.using_activation:
                     self.output_img[h][w]=self.ReLU(value+self.filter.bias)
                else:
                    self.output_img[h][w]=value+self.filter.bias
        if self.maxpool:
            self.produce_max_pooling()
        else:
            self.draw()

    def ReLU(self,x):
        return max(0,x)

    def produce_max_pooling(self): # essentially reduces the image size whilst still keeping the most prominent features
        for h in range(self.maxpool_img_h):
            for w in range(self.maxpool_img_w):
                value=-np.inf

                indexH=0
                indexW=0
                for fh in range(self.maxpoolH):
                    for fw in range(self.maxpoolW):

                        H=h*self.maxpoolStride+fh
                        W=w*self.maxpoolStride+fw
                        val=self.output_img[H][W]
                        if val>value:
                            value=val
                            indexH= H
                            indexW= W
               
                self.maxpool_img[h][w]=value
                self.maxpool_img_indexes[h][w]=[indexH,indexW] # would use x and y but to keep things consistent we are using height and width
        self.drawMaxP()

    def draw(self):
     

        pygame.draw.rect(win,(255,255,255),(self.input_img.tlx,self.input_img.tly,imgw*scale_fac,imgh*scale_fac))

        normalized_arr=normalize_array(self.output_img)

        for y in range(self.height):
            for x in range(self.width):
               
                pix=normalized_arr[y][x]*255
                pix_x=self.input_img.tlx+x*scale_fac #pixel top left corner co - ordinate
                pix_y=self.input_img.tly+y*scale_fac

                pygame.draw.rect(win,(pix,pix,pix),(pix_x,pix_y,scale_fac,scale_fac))

        win.blit(self.text,(self.input_img.tlx,self.input_img.tly-img_pad_y/1.5)) # add image label eg input image, feature map 1 etc
    def drawMaxP(self):
        pygame.draw.rect(win,(255,255,255),(self.input_img.tlx,self.input_img.tly,imgw*scale_fac,imgh*scale_fac))

        adjusted_scale_factor=scale_fac*2
        normalized_arr=normalize_array(self.maxpool_img)

        for y in range(self.maxpool_img_h):
            for x in range(self.maxpool_img_w):
               
                pix=normalized_arr[y][x]*255
                pix_x=self.input_img.tlx+x*adjusted_scale_factor #pixel top left corner co - ordinate
                pix_y=self.input_img.tly+y*adjusted_scale_factor

                pygame.draw.rect(win,(pix,pix,pix),(pix_x,pix_y,adjusted_scale_factor,adjusted_scale_factor))

        win.blit(self.text,(self.input_img.tlx,self.input_img.tly-img_pad_y/1.5)) # add image label eg input image, feature map 1 etc
                   
class filter_input:
    def __init__(self,fil,index=4):
        self.filter=fil  

        self.shape=self.filter.filter.shape

        self.index=index


        self.tlx=0 #top left rect corner
        self.tly=0
        adj_index=self.index//2
        self.tly=img_pad_y*(adj_index+1)+imgh*(adj_index)*scale_fac

        if self.index % 2 ==0:
            self.tlx=img_pad_x
           
        else:
            self.tlx=img_pad_x*2+imgw*scale_fac

        self.pixel_size=scale_fac*7

        self.label="filter"

        self.text_colour=(222, 215, 202)
        self.text=font.render(self.label,True,self.text_colour)

        self.text_colour2=(173, 203, 220)
       

    def draw(self):
        pygame.draw.rect(win,lighter_bg_colour,(self.tlx,self.tly,imgw*scale_fac,imgh*scale_fac))

        pygame.draw.rect(win,(255,255,255),(self.tlx,self.tly,len(self.filter.filter[0])*self.pixel_size,len(self.filter.filter)*self.pixel_size))

        normalized_arr=normalize_array(self.filter.filter)

        for y in range(len(self.filter.filter)):
            for x in range(len(self.filter.filter[0])):
                pix=normalized_arr[y][x]*255
                pix_x=self.tlx+x*self.pixel_size #pixel top left corner co - ordinate
                pix_y=self.tly+y*self.pixel_size

                pygame.draw.rect(win,(pix,pix,pix),(pix_x,pix_y,self.pixel_size,self.pixel_size))
                txtcol=255 if pix<127 else 0
                txt=small_font.render(f"{self.filter.filter[y][x]:.1f}",True,(txtcol,txtcol,txtcol))
                win.blit(txt,(pix_x+self.pixel_size/8,pix_y+self.pixel_size/8))

                biastxt=font.render(f"filter bias: {self.filter.bias:.2f}",True,self.text_colour2)
                win.blit(biastxt,(self.tlx,self.tly+imgh*scale_fac))


        win.blit(self.text,(self.tlx,self.tly-img_pad_y/1.5)) # add image label eg input image, feature map 1 etc

def normalize_array(arr):
    arr2=arr-arr.min()
    m=arr2.max()
    if m==0:
        return arr
    else:
        return arr2/m

     

   

       

   
       

imgs=[img(index=i) for i in range(imgcountW*imgcountH)]
fil=filter3x3()
cimgs=[conv_img(imgs[i+1],fil,False if  i==2 else True,False if i==1 or i==2 else True) for i in range(3)]


filinp=filter_input(fil)

count=0
run=True    
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False

    win.fill(bg_colour)

    if count%1==0:
        rate=0.1
        fil.bias+=0.001
        fil.filter+= np.array([
            [rate/2-random.random()*rate,rate/2-random.random()*rate,rate/2-random.random()*rate],
            [rate/2-random.random()*rate,rate/2-random.random()*rate,rate/2-random.random()*rate],
            [rate/2-random.random()*rate,rate/2-random.random()*rate,rate/2-random.random()*rate]],dtype=np.float16)
        # fil.filter+= np.array([
        #      [rate,rate,rate],
        #      [rate,rate,rate],
        #      [rate,rate,rate]],dtype=np.float16)

    imgs[0].draw()
    for cimg in cimgs:
        cimg.produce_image()

    filinp.draw()

    pygame.display.flip()
    count+=1

