import numpy as np
from PIL import Image
import pygame
import random

pygame.init()
font=pygame.font.Font("SpaceMono-Bold.ttf",15)

#allowing multiple images (input images, feature maps, predicted images) for better visualization

imgcountH=2
imgcountW=2

img_pad_x=40
img_pad_y=40

imgw=28
imgh=28

scale_fac=5

win_w=imgcountW*imgw*scale_fac+img_pad_x*(imgcountW+1)
win_h=imgcountH*imgh*scale_fac+img_pad_y*(imgcountH+1)

win=pygame.display.set_mode((win_w,win_h))
bg_colour=(22, 23, 26)
pygame.display.set_caption("CNN - recognizing digits")
icon = pygame.image.load(r"media\test_img.ico").convert_alpha()
pygame.display.set_icon(icon)
# convolutional neural net

class filter3x3: # the kernel that slides over the image to create a feature map
    def __init__(self):
        self.filter=np.array([
            [0,0,0],
            [0,0,0],
            [0,0,0]],dtype=np.float16) # numbers represent weights to multiply the pixel value by
         
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
        self.tly=img_pad_y*(adj_index+1)+imgh*(adj_index)*scale_fac


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
    def __init__(self,image,fil):
        self.input_img=image
        self.filter=fil
        self.stride=1

        self.filterw=3
        self.filterh=3

        self.padx=1
        self.pady=1

        self.padded= np.pad(self.input_img.img,((self.pady,self.pady),(self.padx,self.padx)),mode='constant',constant_values=0)

        self.width=int( (self.input_img.width + 2*self.padx-self.filterw)/self.stride + 1)
        self.height=int( (self.input_img.height + 2*self.pady-self.filterh)/self.stride + 1)

        self.output_img=np.array([[0 for _ in range(self.width)] for _ in range(self.height)],dtype=np.float16)

        self.label="max pooled"

        self.text_colour=(222, 215, 202)
        self.text=font.render(self.label,True,self.text_colour)

        self.maxpool=True
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
                self.output_img[h][w]=value
        if self.maxpool:
            self.produce_max_pooling()
        else:
            self.draw()

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

        for y in range(self.height):
            for x in range(self.width):
                
                pix=max(0,self.output_img[y][x]/(self.output_img.max()+0.001))*255
                pix_x=self.input_img.tlx+x*scale_fac #pixel top left corner co - ordinate
                pix_y=self.input_img.tly+y*scale_fac

                pygame.draw.rect(win,(pix,pix,pix),(pix_x,pix_y,scale_fac,scale_fac))


        win.blit(self.text,(self.input_img.tlx,self.input_img.tly-img_pad_y/1.5)) # add image label eg input image, feature map 1 etc
    def drawMaxP(self):
        pygame.draw.rect(win,(255,255,255),(self.input_img.tlx,self.input_img.tly,imgw*scale_fac,imgh*scale_fac))

        adjusted_scale_factor=scale_fac*2

        for y in range(self.maxpool_img_h):
            for x in range(self.maxpool_img_w):
                
                pix=max(0,self.maxpool_img[y][x]/(self.maxpool_img.max()+0.001))*255
                pix_x=self.input_img.tlx+x*adjusted_scale_factor #pixel top left corner co - ordinate
                pix_y=self.input_img.tly+y*adjusted_scale_factor

                pygame.draw.rect(win,(pix,pix,pix),(pix_x,pix_y,adjusted_scale_factor,adjusted_scale_factor))


        win.blit(self.text,(self.input_img.tlx,self.input_img.tly-img_pad_y/1.5)) # add image label eg input image, feature map 1 etc
                    

          
        


    
        

imgs=[img(index=i) for i in range(imgcountW*imgcountH)]
fils=[filter3x3() for _ in range(3)]
cimgs=[conv_img(imgs[i+1],fils[i]) for i in range(3)]

cimgs[1].maxpool=False
cimgs[1].label="feature map"
cimgs[1].text=font.render(cimgs[1].label,True,cimgs[1].text_colour)



count=0
run=True     
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False

    win.fill(bg_colour)

    if count%10==0:
        fils[random.randint(0,2)].filter= np.array([
            [5-random.random()*10,5-random.random()*10,5-random.random()*10],
            [5-random.random()*10,5-random.random()*10,5-random.random()*10],
            [5-random.random()*10,5-random.random()*10,5-random.random()*10]],dtype=np.float16)


    imgs[0].draw()
    for cimg in cimgs:
        cimg.produce_image()

    pygame.display.flip()
    count+=1
