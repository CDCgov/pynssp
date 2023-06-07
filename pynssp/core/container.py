class NSSPContainer:
  """A NSSPContainer Class to store a value or an object

    An object of class NSSPContainer stores a value or an object
    The NSSPContainer class encapsulates a value or an object

    :param value: value to store
    :ivar value: Stored value
    :examples:

      >>> from pynssp import NSSPContainer
      >>> 
      >>> cont = NSSPContainer("abcdef")
  """

  def __init__(self, value):
    """Initializes an NSSPContainer class
    """
    self.value = value


class APIGraph:
  """A class to store an API graph

    :param path: a string representing the location of a graph
    :param response: an object of class response
    :ivar path: the location of a graph file
    :ivar response: a response object
  """

  def __init__(self, path, response):
    """Initializes an APIGraph class
    """
    self.path = path
    self.response = response


  def __str__(self):
    """Print an API object"""
    return f"{self.path}"


  def show(self):
    """A method to display an APIGraph object"""
    from PIL import Image
    img = Image.open(self.path)
    img.show()


  def plot(self):
    """A method to plot an APIGraph object"""
    from skimage.io import imread, imshow
    img = imread(self.path)
    imshow(img)
