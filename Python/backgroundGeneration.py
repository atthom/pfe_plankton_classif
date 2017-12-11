# imports
from PIL import Image
from numpy import *

def main():
    size = (416, 416)
    img = Image.new('RGB', size, (255,255,255))
    img.save("background416_416.png", 'png')
main()
print("\n\ndone.")
