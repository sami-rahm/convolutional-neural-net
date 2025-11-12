from PIL import Image,ImageDraw,ImageFont

font=ImageFont.truetype("SpaceMono-Bold.ttf",size=20)
WIDTH=20
HEIGHT=20

img=Image.new(mode="L",size=(WIDTH,HEIGHT),color=0)
#img.resize((WIDTH,HEIGHT),resample=Image.NEAREST)

num=1
draw=ImageDraw.Draw(img)
draw.text((7,0),str(num),(255),font=font)

img.save("media/test_img1.png")
img.show()