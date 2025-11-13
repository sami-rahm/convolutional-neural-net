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
        
class img:
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

        pygame.draw.rect(win,(255,255,255),(self.tlx,self.tly,28*5,28*5))

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

        self.padx=0
        self.pady=0

        self.width=int( (self.input_img.width + 2*self.padx-self.filterw)/self.stride + 1)
        self.height=int( (self.input_img.height + 2*self.pady-self.filterh)/self.stride + 1)

        self.output_img=np.array([[0 for _ in range(self.width)] for _ in range(self.height)],dtype=np.float16)

        self.label="feature map"

        self.text_colour=(222, 215, 202)
        self.text=font.render(self.label,True,self.text_colour)

    def produce_image(self):
        for h in range(self.height):
            for w in range(self.width):
                value=0
                for fh in range(self.filterh):
                    for fw in range(self.filterw):
                        value+=self.filter.filter[fh][fw]*self.input_img.img[h*self.stride+fh][w*self.stride+fw]
                #print(value)
                self.output_img[h][w]=value
        self.draw()

    def draw(self):

        pygame.draw.rect(win,(255,255,255),(self.input_img.tlx,self.input_img.tly,28*5,28*5))

        for y in range(self.height):
            for x in range(self.width):
                
                pix=max(0,self.output_img[y][x]/(self.output_img.max()+0.001))*255
                pix_x=self.input_img.tlx+x*scale_fac #pixel top left corner co - ordinate
                pix_y=self.input_img.tly+y*scale_fac

                pygame.draw.rect(win,(pix,pix,pix),(pix_x,pix_y,scale_fac,scale_fac))


        win.blit(self.text,(self.input_img.tlx,self.input_img.tly-img_pad_y/1.5)) # add image label eg input image, feature map 1 etc
                    

          
        


    
        

imgs=[img(index=i) for i in range(imgcountW*imgcountH)]
fils=[filter3x3() for _ in range(3)]
cimgs=[conv_img(imgs[i+1],fils[i]) for i in range(3)]



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
