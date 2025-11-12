import numpy as np
from PIL import Image
import pygame

pygame.init()
font=pygame.font.Font("SpaceMono-Bold.ttf",15)

#allowing multiple images (input images, feature maps, predicted images) for better visualization

imgcountH=3
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
            [1,0,1],
            [1,0,1],
            [1,0,1]],dtype=np.float16) # numbers represent weights to multiply the pixel value by
        
class img:
    def __init__(self,filepath=r"media\test_img.png",index=1): # index represents the position in the window (top left to bottom right)
        
        self.filepath=filepath 

        self.img=np.array(Image.open(filepath),dtype=np.float16)/255 if filepath is not None else None# normalized image values

        self.height,self.width=self.img.shape # image height, width

        self.index=index

        self.label="input image"

        self.tlx=0 #top left rect corner
        self.tly=0
        adj_index=int(self.index/2)
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
    def __init__(self):
        pass        
        


    
        

imgs=[img(index=i) for i in range(imgcountW*imgcountH)]

run=True     
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False

    win.fill(bg_colour)
    for img in imgs: #draw all images
        img.draw()

    pygame.display.flip()