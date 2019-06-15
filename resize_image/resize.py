#Ram Bhattarai
#Credit to Sparsha Saha
#Ming Ma
#Zhe Lu
#ECE 544
#Resize the image




from PIL import Image

#Change the actual filename
#Change the actual foldername
#Change the how many images you want to resize
foldername = "peace_test/"
filename =   "peace_"
size     = 1000


#Change the Your computer full path to your actual path of the computer
path = "/Your Computer Full Path/"+str(foldername)+str(filename);

#Function to resize the image
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)


for i in range(0, size):
    # Mention the directory in which you wanna resize the images followed by the image name
    resizeImage(path + str(i) + '.png')
